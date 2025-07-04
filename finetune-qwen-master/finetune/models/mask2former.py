import pickle
import types

import torch
import torch.nn as nn
from fvcore.common.config import CfgNode

from .segmentation.mask2former.Mask2Former_Simplify import (
    MSDeformAttnPixelDecoder,
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
)
from .segmentation.mask2former.mask_config.config import Config
from .segmentation.mask2former.swin_trans import SWIN_BUILDERS


class MLPWithUpsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(2, 16)  # 输入2，输出16
        self.mask_sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
        x = self.mlp(x)  # [B, H, W, 16]
        x = x.permute(0, 3, 1, 2)  # [B, 16, H, W]
        x = x.reshape(B, 1, H * 4, W * 4)  # [B, 1, 512, 512]
        return self.mask_sigmoid(x)


class MaskModel(nn.Module):
    def __init__(self, hidden_size, mask_config, pretrain_path):
        super().__init__()  # 加入这行，初始化父类
        self.hidden_size = hidden_size
        self.mask_decoder_cfg = self.get_mask_config(mask_config)
        self.seg_query = nn.Parameter(
            torch.zeros([self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, self.hidden_size])
        )
        self.seg_query_projector = nn.Linear(self.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.class_name_projector = nn.Linear(self.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.num_queries = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES

        swin_type = getattr(self.mask_decoder_cfg, "swin_type", "base")
        self.vision_tower_mask = SWIN_BUILDERS[swin_type]

        # 禁止self.vision_tower_mask参与梯度计算
        for param in self.vision_tower_mask.parameters():
            param.requires_grad = False

        self.load_pretrain(pretrain_path)

    @classmethod
    def get_mask_config(cls, config):
        cfg_coco = Config.fromfile(config)
        cfg_base = CfgNode.load_yaml_with_base(config, allow_unsafe=True)
        cfg_base.update(cfg_coco.__dict__.items())
        cfg = cfg_base
        cfg = Config(cfg)
        return cfg

    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [
            int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2**i)
            for i in range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))
        ]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }

        # 转为支持点访问的版本
        backbone_feature_shape = {
            name: types.SimpleNamespace(channel=out_feature_channels[name], stride=out_feature_strides[name])
            for name in out_features
        }
        return backbone_feature_shape

    def pixel_decoder_init(self, cfg):
        input_shape = self.output_shape()
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = (
            cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        )  # ["res3", "res4", "res5"]

        return MSDeformAttnPixelDecoder(
            input_shape,
            transformer_dropout,
            transformer_nheads,
            transformer_dim_feedforward,
            transformer_enc_layers,
            conv_dim,
            mask_dim,
            transformer_in_features,
            common_stride,
        )

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE
        seg_concat = False
        print(
            f"current seg concat mode: {seg_concat}, seg_norm: {seg_norm}, seg_proj: {seg_proj}, seg_fuse_score: {seg_fuse_score}"
        )
        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(
            in_channels,
            hidden_dim,
            num_queries,
            nheads,
            dim_feedforward,
            dec_layers,
            pre_norm,
            mask_dim,
            enforce_input_project,
            seg_norm,
            seg_concat,
            seg_proj,
            seg_fuse_score,
        )
        return predictor

    def load_pretrain(self, pretrained_path):
        if pretrained_path is None:
            return

        def get_w(weights, keyword):
            return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

        def change_w(weights, old_name, new_name):
            weights[new_name] = weights[old_name]
            weights.pop(old_name)

        if pretrained_path.endswith(".pkl"):
            with open(pretrained_path, "rb") as f:
                ckpt = pickle.load(f)
        else:
            ckpt = torch.load(pretrained_path)
        pixel_decoder_weights = get_w(ckpt["model"], "sem_seg_head.pixel_decoder")
        predictor_weights = get_w(ckpt["model"], "sem_seg_head.predictor")
        pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
        predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

        # deal some diff keys
        change_w(pixel_decoder_weights, "adapter_1.weight", "adapter_1.0.weight")
        change_w(pixel_decoder_weights, "adapter_1.norm.weight", "adapter_1.1.weight")
        change_w(pixel_decoder_weights, "adapter_1.norm.bias", "adapter_1.1.bias")
        change_w(pixel_decoder_weights, "layer_1.weight", "layer_1.0.weight")
        change_w(pixel_decoder_weights, "layer_1.norm.weight", "layer_1.1.weight")
        change_w(pixel_decoder_weights, "layer_1.norm.bias", "layer_1.1.bias")
        if "static_query.weight" in predictor_weights:
            change_w(predictor_weights, "static_query.weight", "query_feat.weight")
        if predictor_weights["query_embed.weight"].shape[0] == 200:
            predictor_weights["query_embed.weight"] = predictor_weights["query_embed.weight"][:100, :]
        diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights, strict=False)

        # 修改predictor_weights中的相关参数维度
        predictor_weights["query_feat.weight"] = predictor_weights["query_feat.weight"][: self.num_queries, :]
        predictor_weights["query_embed.weight"] = predictor_weights["query_embed.weight"][: self.num_queries, :]
        diff_predictor_msg = self.predictor.load_state_dict(predictor_weights, strict=False)

        print(diff_predictor_msg)
        print(diff_pixel_msg)

    def get_vision_tower_feature(self, images):
        features = self.vision_tower_mask(images)
        return {
            "res2": features[0],  # bs, 128, 256, 256
            "res3": features[1],  # bs, 256, 128, 128
            "res4": features[2],  # bs, 512, 64, 64
            "res5": features[3],  # bs, 1024, 32, 32
        }

    def forward(self, seg_query, class_name_embedding, image, SEG_EMBEDDING=None, REGIN_EMBEDDING_LIST=None):
        """
        seg_query: 模型分割token给出的内容
        class_name_embedding: 类别名称的embedding
        """

        seg_query = self.seg_query_projector(seg_query.to(dtype=torch.float32))
        class_name_embedding = self.class_name_projector(class_name_embedding.to(dtype=torch.float32))

        with torch.no_grad():
            image_features = self.get_vision_tower_feature(image)

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features
        )
        mask_outputs = self.predictor(
            multi_scale_features,  # 由图像决定
            mask_features,  # 由图像决定
            None,  # mask
            seg_query,
            SEG_EMBEDDING,  # SEG_EMBEDDING
            class_name_embedding,  # class_name_embedding
            REGIN_EMBEDDING_LIST,  # REGIN_EMBEDDING_LIST
            using_box=False,
            using_boxiou=False,
            return_hs=False,
        )
        return mask_outputs
