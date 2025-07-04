from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sam2.build_sam import build_sam2_hf
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor


class MLPSparse(nn.Module):
    """
    # 目标:
        [B, n, 2048] -> [B, 3, 256]
    简化版本，不使用 Dropout，且降低 fc1 的输出维度
    """

    def __init__(self, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 2048 -> 1024
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim * 3),  # 1024 -> 768
        )

    def forward(self, x):
        x = self.net(x)  # [B, n, 768]
        x = x.view(x.shape[0], 3, -1)  # reshape为 [B, 3, 256]
        return x


class MLPDense(nn.Module):
    """
    # 目标:
    [B, 32, hidden_dim] -> [B, 256, 64, 64]
    """

    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.proj1 = nn.Linear(hidden_dim, 512)
        self.act1 = nn.GELU()
        # 将生成的特征图从 [B, 4, 8, 512] 重塑后，使用 PixelShuffle
        # 需要把通道数翻倍以适应 PixelShuffle 的 r^2 倍
        self.conv_pixelshuffle = nn.Sequential(
            nn.Conv2d(512, 512 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 输出通道: 512, 尺寸翻倍
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 再次上采样
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # 额外的上采样层，视初始尺寸而定
        self.extra_upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, x):
        if x is None:
            return None
        B, N, D = x.shape  # [B, 32, hidden_dim]
        x = self.proj1(x)  # [B, 32, 512]
        x = self.act1(x)
        x = x.view(B, 4, 8, 512).permute(0, 3, 1, 2)  # [B, 512, 4, 8]
        x = self.conv_pixelshuffle(x)
        x = self.extra_upsample(x)
        x = self.proj2(x)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=True)
        return x


class MLPClsHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # [B, 256]
            nn.Linear(in_dim, hidden_dim),  # 256 -> hidden_dim
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # hidden_dim -> 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 256]，这里 B=3
        # 先对第1维 (n=1) 以及 batch 维度一起平均，得到 [256]
        feat: torch.Tensor = x.mean(dim=(0, 1))  # [256]
        # 送进 MLP，unsqueeze 扩成 [1,256]
        logits = self.net(feat.unsqueeze(0))  # [1,1]
        # 返回一个值
        return logits.squeeze()


class SAM2Predictor(nn.Module):
    def __init__(self, model_name=None, hidden_size=2048, is_finetune=False):
        """
        两个
            阶段1: 微调SAM的prompt encoder和mask decoder
            阶段2: 大模型生成的输出微调SAM的mask decoder
        """
        super().__init__()
        # Load the SAM2 model
        if not model_name:
            model_name = "facebook/sam2.1-hiera-base-plus"
            logger.warning(
                f"No model name provided. Using default model: {model_name} "
                "Please specify a model name if you want to use a different one."
            )
        self.model: SAM2Base = build_sam2_hf(model_name)
        self.sam2_image_predictor = SAM2ImagePredictor(self.model)

        self.model_name = model_name
        self.device = self.model.device
        self.hidden_size = hidden_size
        self.is_finetune = is_finetune

        # 冻结image encoder
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        if not self.is_finetune:
            logger.info("SAM2Predictor: 实现与大模型的联合微调")
            self.mlp_dense = MLPDense(hidden_dim=hidden_size)
            self.mlp_sparse = MLPSparse(hidden_dim=hidden_size, out_dim=256)
            self.mlp_cls = MLPClsHead(in_dim=256, hidden_dim=128)
            logger.info("SAM2Predictor: 仅开放mask decoder进行微调")
            for param in self.model.sam_mask_decoder.parameters():
                param.requires_grad = True

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # 动态加载权重
        if "state_dict" in state_dict:
            state = state_dict["state_dict"]
        else:
            state = state_dict
        # 动态处理 "module." 前缀
        if any(k.startswith("module.") for k in state):
            processed = {k[len("module.") :]: v for k, v in state.items()}
        else:
            processed = state

        # 过滤掉shape不匹配的参数
        model_state = self.state_dict()
        filtered = {}
        mismatched = []
        for k, v in processed.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                mismatched.append(k)
        if mismatched:
            logger.warning(f"以下权重因shape不匹配未被载入: {mismatched}")

        return super().load_state_dict(filtered, strict, assign)

    def get_image_feat(self, image: torch.Tensor) -> dict:
        input_image = self.sam2_image_predictor._transforms.transforms(image)
        input_image = input_image[None, ...].to(self.device)

        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        return feats[-1], feats[:-1]

    def combine_embedding(self, dense_embedding=None, sparse_embedding=None, box=None, img_idx=-1):
        unnorm_box = self.sam2_image_predictor._prep_prompts(
            point_coords=None,
            point_labels=None,
            box=box,
            mask_logits=None,
            normalize_coords=True,
            img_idx=img_idx,
        )[-1]

        if unnorm_box is not None and unnorm_box.numel() != 0:
            # 对 box 进行处理并生成对应的点信息
            box_coords = unnorm_box.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
            box_labels = box_labels.repeat(unnorm_box.size(0), 1)
            concat_points = (box_coords, box_labels)
        else:
            concat_points = None

        # 使用 SAM prompt encoder 对 box 进行编码
        new_sparse, new_dense = self.model.sam_prompt_encoder(points=concat_points, boxes=None, masks=None)

        if dense_embedding is not None and dense_embedding.shape[0] > 0:
            dense_embedding = dense_embedding.unsqueeze(0)
            dense_embedding = (
                torch.cat([dense_embedding, new_dense], dim=0) if dense_embedding is not None else new_dense
            )
        else:
            dense_embedding = new_dense

        if sparse_embedding is not None and sparse_embedding.shape[0] > 0:
            sparse_embedding = sparse_embedding.unsqueeze(0)
            sparse_embedding = (
                torch.cat([sparse_embedding, new_sparse], dim=0) if sparse_embedding is not None else new_sparse
            )
        else:
            sparse_embedding = new_sparse

        return sparse_embedding, dense_embedding

    def forward(self, image, mask_token_feat=None, point_token_feat=None, boxes=None):
        """
        1. 图像mask特征直接作为dense_embeddings的特征
        2. 图像点的相关输出直接作为sparse_embeddings的特征

        # 待实验:
            1. 加入<bbox>标签
            2. 加入box
            3. 二者都加入
        """
        # 设置图像 batch 并获取特征
        self.sam2_image_predictor.set_image_batch(image)
        feat = self.sam2_image_predictor._features
        image_embeds, high_res_feats = feat["image_embed"], feat["high_res_feats"]

        bs = len(image)

        # 联调后, 使用 mlp_dense 与 mlp_sparse 处理 token 特征
        if not self.is_finetune:
            dense_embeddings = self.mlp_dense(mask_token_feat)
            if dense_embeddings is None:
                dense_embeddings = torch.empty((bs, 0, 64, 64), device=self.device)
            if point_token_feat is None:
                sparse_embeddings = torch.empty((bs, 0, 256), device=self.device)
            else:
                sparse_embeddings = self.mlp_sparse(point_token_feat)

        masks, logits = [], []
        for i in range(bs):
            box = boxes[i] if boxes else None
            if not self.is_finetune:
                # 大模型联合微调时: 传入预计算的 dense 与 sparse embedding（取 batch 中第一个）
                sparse_embedding, dense_embedding = self.combine_embedding(
                    dense_embeddings[i],
                    sparse_embeddings[i],
                    box,
                    i,
                )
            else:
                # 单独训练时: 训练时不使用 mask_token_feat 的预计算 embedding，而直接用 box 信息生成 embedding
                sparse_embedding, dense_embedding = self.combine_embedding(box=box, img_idx=i)

            high_res_feature = [feat_level[i].unsqueeze(0) for feat_level in high_res_feats]
            low_res_masks, iou_predictions, sam_tokens_out, object_score_logits = self.model.sam_mask_decoder(
                image_embeddings=image_embeds[i].unsqueeze(0),  # 图像编码
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),  # 位置编码
                sparse_prompt_embeddings=sparse_embedding,
                dense_prompt_embeddings=dense_embedding,
                high_res_features=high_res_feature,  # 高质量图像编码
                multimask_output=False,
                repeat_image=True,
            )
            mask = self.sam2_image_predictor._transforms.postprocess_masks(
                low_res_masks, self.sam2_image_predictor._orig_hw[i]
            )
            low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
            masks.append(mask.mean(dim=0, keepdim=True).squeeze(0))
            logit = self.mlp_cls(sam_tokens_out)
            logits.append(logit)
        return torch.stack(masks, dim=1).squeeze(0), torch.stack(logits)  # [batch, 1, 64, 64]
