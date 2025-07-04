import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from swift.llm import TemplateType
from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_vl
from swift.llm.model.register import Model, ModelGroup, ModelMeta, register_model
from swift.llm.model.utils import HfConfigFactory
from transformers import PreTrainedTokenizer, Qwen2_5_VLForConditionalGeneration

from ..utils.constants import (
    DEFAULT_BOX_END_TOKEN,
    DEFAULT_BOX_START_TOKEN,
    DEFAULT_CLASS_TOKEN,
    DEFAULT_EXPERT_TOKEN,
    DEFAULT_MASK_TOKEN,
    DEFAULT_POS_TOKEN,
    DEFAULT_RES_END_TOKEN,
    DEFAULT_RES_START_TOKEN,
    IGNORE_INDEX,
    ForensicModelType,
)
from .experts import NPRModel
from .head import ExpertToLLM, MLPClsHead, MLPMaskHead
from .loss import DiceLoss, WeightedCrossEntropyWithLabelSmoothing
from .model_output import AlignLLMOutputWithPast
from .sam2_predictor import SAM2Predictor


class QwenForensicModel(Qwen2_5_VLForConditionalGeneration):
    def __init__(
        self,
        config,
        num_new_tokens,
        text_weight,
        cls_weight,
        bce_weight,
        dice_weight,
        add_expert_feat=False,
        add_mask_predict=True,
        add_cls_predict=True,
        resize_image_size=1024,
        expert_model_weight_path=None,
        **kwargs,
    ):
        super().__init__(config)
        # 损失函数设置
        self.bce_weight = bce_weight
        self.text_weight = text_weight
        self.loss_cls = WeightedCrossEntropyWithLabelSmoothing(loss_weight=cls_weight, use_sigmoid=True)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_dice = DiceLoss(loss_weight=dice_weight, use_sigmoid=False)
        self.loss_text = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self.hidden_size = self.config.hidden_size
        self.num_new_tokens = num_new_tokens
        self.add_expert_feat = add_expert_feat
        self.add_mask_predict = add_mask_predict
        self.add_cls_predict = add_cls_predict
        self.resize_image_size = resize_image_size

        # 写入配置文件
        self.config.image_size = resize_image_size
        self.config.num_new_tokens = num_new_tokens
        self.config.add_expert_feat = add_expert_feat
        self.expert_model_weight_path = expert_model_weight_path

    # inference 3.with expert model
    def _init_expert_model(self):
        # 载入专家模型
        self.expert_model = NPRModel()
        # 载入模型权重
        target_module = torch.load(self.expert_model_weight_path, weights_only=False)["model"]
        new_state_dict = {k.replace("model.", "module."): v for k, v in target_module.items()}
        self.expert_model.load_state_dict(new_state_dict, strict=True)
        # 转移到和当前模型相同的设备上
        self.expert_model.to(self.device, dtype=self.model.dtype)
        self.expert_model.eval()

    def get_expert_feat(self, img_clips):
        if isinstance(img_clips, list):
            # 如果是list，转为np.array
            img_clips = np.stack(img_clips, axis=0)

        if isinstance(img_clips, np.ndarray):
            # 如果是[B, H, W, C]，需转为[B, C, H, W]
            if img_clips.ndim == 4 and img_clips.shape[-1] in [1, 3]:
                img_clips = np.transpose(img_clips, (0, 3, 1, 2))

        # 保证数据和模型在同一设备和数据类型
        if isinstance(img_clips, np.ndarray):
            img_clips = torch.from_numpy(img_clips).float()  # 转为float tensor

        img_clips = img_clips.to(self.device, dtype=self.model.dtype)

        if self.expert_model is None:
            logger.info("初始化专家模型...")
            self._init_expert_model()

        # 确保img_clips和trufor模型的数据类型一致
        if len(img_clips.shape) == 4:  # [batch, channels, height, width]
            expert_feature = self.expert_model.forward(img_clips)
        else:  # 处理单张图片
            # 解决单张图片时的维度问题
            expert_feature = self.expert_model.forward(img_clips.unsqueeze(0))

        # 要求进行混合精度计算
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            result = self.expert_head(expert_feature)
        return result

    def get_tokens_for_customization(self):
        # 增加获取自定义token的功能
        new_token_id = self.detection_ids + self.mask_ids
        if self.position_ids:
            new_token_id += self.position_ids
        if self.add_expert_feat and self.expert_ids:
            new_token_id += self.expert_ids
        ret = new_token_id + [self.tokenizer.eos_token_id]
        return ret

    def setup_tokens_for_conversation(self, tokenizer: PreTrainedTokenizer, sam_model_path=None):
        pad_token = tokenizer.pad_token_id
        if pad_token is None:
            pad_token = tokenizer.eos_token_id
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = pad_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = pad_token
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None

        self.tokenizer: PreTrainedTokenizer = tokenizer
        # <mask_0>... <mask_num_new_tokens>
        mask_tokens = [DEFAULT_MASK_TOKEN.format(i=i) for i in range(self.num_new_tokens)]
        detection_tokens = [DEFAULT_CLASS_TOKEN]
        position_tokens = [DEFAULT_POS_TOKEN]
        self.tokenizer.add_tokens(mask_tokens + detection_tokens + position_tokens, special_tokens=True)

        # 加入占位标签
        self.placeholder_tokens = [
            DEFAULT_BOX_START_TOKEN,
            DEFAULT_BOX_END_TOKEN,
            DEFAULT_RES_START_TOKEN,
            DEFAULT_RES_END_TOKEN,
        ]
        self.tokenizer.add_tokens(self.placeholder_tokens, special_tokens=True)

        # 加入专家token
        if self.add_expert_feat:
            expert_tokens = [DEFAULT_EXPERT_TOKEN.format(i=i) for i in range(self.num_new_tokens)]
            self.tokenizer.add_tokens(expert_tokens, special_tokens=True)
            self.expert_ids = self.tokenizer.convert_tokens_to_ids(expert_tokens)
        self.mask_ids = self.tokenizer.convert_tokens_to_ids(mask_tokens)
        self.detection_ids = self.tokenizer.convert_tokens_to_ids(detection_tokens)
        self.position_ids = self.tokenizer.convert_tokens_to_ids(position_tokens)
        self.placeholder_ids = self.tokenizer.convert_tokens_to_ids(self.placeholder_tokens)

        # 获取resize后的新嵌入层
        new_embeddings = self.resize_token_embeddings(len(self.tokenizer))

        # 初始化新token的embedding权重
        num_new_tokens = len(mask_tokens) + len(detection_tokens) + len(position_tokens)

        # 基于现有embedding的均值和标准差初始化
        existing_embeds = new_embeddings.weight[:-num_new_tokens].data
        mean, std = existing_embeds.mean(dim=0), existing_embeds.std(dim=0)
        new_tokens_weight = torch.randn_like(new_embeddings.weight[-num_new_tokens:].data)
        new_tokens_weight = new_tokens_weight * std + mean
        new_embeddings.weight[-num_new_tokens:].data.copy_(new_tokens_weight)

        # 确保梯度跟踪
        new_embeddings.weight = nn.Parameter(new_embeddings.weight.data, requires_grad=True)
        self.set_input_embeddings(new_embeddings)

        # 配置条目调整
        pad_token_id = self.model.config.pad_token_id or tokenizer.pad_token_id
        HfConfigFactory.set_model_config_attr(self.model, "pad_token_id", pad_token_id)

        # 初始化lm_head
        # 重新初始化 lm_head 可以确保投影层的维度和权重更符合新任务的要求。
        new_vocab_size = len(self.tokenizer)
        num_new_tokens = self.num_new_tokens  # 新增token数量

        # 创建新的 lm_head，其输出维度使用新的词汇表大小
        new_lm_head = nn.Linear(self.model.config.hidden_size, new_vocab_size, bias=False)

        # 基于现有embedding的均值和标准差为新增token初始化权重
        new_token_weight = torch.randn(num_new_tokens, self.model.config.hidden_size, device=mean.device)
        new_token_weight = new_token_weight * std + mean

        # 复制原lm_head中已有的权重
        # 新lm_head的前部分应为旧权重，但如果旧权重的行数多余需要，则只复制前num_existing行
        num_existing = new_vocab_size - num_new_tokens
        old_weight = self.lm_head.weight.data
        num_existing_to_copy = min(old_weight.shape[0], num_existing)
        new_lm_head.weight.data[:num_existing_to_copy] = old_weight[:num_existing_to_copy]

        # 如果新设定的num_existing超过了旧权重行数，则额外部分随机初始化
        if num_existing_to_copy < num_existing:
            extra = num_existing - num_existing_to_copy
            rand_weight = torch.randn(extra, self.model.config.hidden_size, device=mean.device) * std + mean
            new_lm_head.weight.data[num_existing_to_copy:num_existing] = rand_weight

        # 将新增token的权重赋值
        new_lm_head.weight.data[num_existing:] = new_token_weight
        new_lm_head.weight.requires_grad = True
        self.lm_head = new_lm_head.to(self.device, dtype=self.model.dtype)

        # 初始化其他头部
        self.cls_head = MLPClsHead(self.hidden_size).to(self.device, dtype=self.model.dtype)
        # self.mask_head = MLPMaskHead(self.num_new_tokens, self.hidden_size, output_size=self.resize_image_size)

        # 加载SAM2模型, 并载入权重
        if sam_model_path and Path(sam_model_path).exists():
            sam2_model = torch.load(sam_model_path, map_location=self.device)
            self.mask_head = SAM2Predictor(sam2_model["model_name"], self.hidden_size).to(self.device, dtype=self.model.dtype)
            self.mask_head.load_state_dict(sam2_model, strict=False)
            logger.info(f"加载SAM模型权重成功: {sam_model_path}")
            logger.info(f"载入SAM权重成功, 细节内容如下")
            # 写入sam2_model除了state_dict以外的参数
            for k, v in sam2_model.items():
                if k not in ["state_dict"]:
                    logger.info(f"{k} : {v}")
        else:
            # 如果没有提供权重文件，则使用默认的SAM2模型
            # 统一由外部加载权重
            self.mask_head = SAM2Predictor(hidden_size=self.hidden_size).to(self.device, dtype=self.model.dtype)

        if self.add_mask_predict is False:
            # 冻结mask_head的全部参数
            for param in self.mask_head.parameters():
                param.requires_grad = False
        if self.add_cls_predict is False:
            # 冻结cls_head的全部参数
            for param in self.cls_head.parameters():
                param.requires_grad = False
        if self.add_expert_feat:
            self.expert_model = None
            self.expert_head = ExpertToLLM(
                in_channels=64,
                out_features=self.hidden_size,
                num_new_tokens=num_new_tokens,
            ).to(self.device)

    def get_embeddings_feature(self, hidden_states, target_token_ids, input_ids):
        """
        根据找到的位置索引，从hidden_states中提取对应的特征向量并返回
        模型生成的 hidden_states 已经在内部整合了位置信息（例如通过 RoPE 或其他位置编码方法），所以每个 token 的隐藏表示已经携带了正确的位置信息。
        通过对 input_ids 进行匹配获得对应的 token 在序列中的 index 后，直接使用这些索引访问 hidden_states 就可获取目标 token 的特征，无需再额外计算或调整偏移量
        """
        if target_token_ids is None:
            raise ValueError("target_token_ids不能为空")

        batch_size, seq_len = input_ids.shape
        device = hidden_states.device
        num_targets = len(target_token_ids)

        # 查找匹配的token id
        target_tensor = torch.as_tensor(target_token_ids, dtype=torch.long, device=device)
        # 转移 input_ids 到 device 上，并增加维度
        mask = input_ids.to(device).unsqueeze(-1) == target_tensor.view(1, 1, num_targets)  # [batch, seq, target]

        # 验证每个target在每个样本中至少出现一次
        match_counts = mask.sum(dim=1)  # [batch, target]
        if (match_counts == 0).any():
            # 构建详细错误信息
            error_mask = match_counts == 0
            error_batches, error_tokens = error_mask.nonzero(as_tuple=True)
            error_dict = defaultdict(list)
            for b, t in zip(error_batches.cpu().tolist(), error_tokens.cpu().tolist()):
                error_dict[b].append(target_tensor[t].item())
            error_msg = "\n".join(
                [f"批次 {b}: token[{tokens}]{self.tokenizer.decode(tokens)} 未匹配到任何位置" for b, tokens in error_dict.items()]
            )
            logger.error(f"目标token匹配失败: {error_msg}")
            return None

        # 如果有重复匹配，则按照最后一个匹配的位置为准
        # 将 mask 沿序列维度翻转，然后获取第一个True对应的索引，转回原来的index
        reversed_mask = mask.flip(dims=[1])  # 翻转序列维度
        _, last_match_reversed_indices = reversed_mask.max(dim=1)
        last_match_indices = seq_len - 1 - last_match_reversed_indices  # [batch, target]

        batch_grid = torch.arange(batch_size, device=device)[:, None].expand(-1, num_targets)  # [batch, target]
        selected = hidden_states[batch_grid, last_match_indices, :]
        return selected

    def get_feature_by_interval(self, hidden_states, start_id, end_id, input_ids):
        batch_size, seq_len = input_ids.shape
        ret_features = []
        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            if start_id not in tokens or end_id not in tokens:
                logger.error(f"批次 {b}: 无法找到start_id[{start_id}]或end_id[{end_id}]")
                continue
            # 查找start_id的首次出现位置
            start_index = tokens.index(start_id)
            # 查找在start_index之后第一次出现end_id的位置
            try:
                end_index = tokens.index(end_id, start_index + 1)
            except ValueError:
                logger.error(f"批次 {b}: 未在start_id之后找到end_id {end_id}")
                continue
            # 提取start_index+1至end_index之间（不含start_id和end_id）的所有特征
            features = hidden_states[b, start_index + 1 : end_index, :]
            ret_features.append(features)

        # 转为tensor
        if len(ret_features) > 0:
            return torch.stack(ret_features, dim=0)
        else:
            return None

    def get_bbox_text_content(self, input_ids: torch.Tensor):
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        pattern = re.compile(r"<\|box_start\|>(.*?)<\|pos\|>")  # pos才是真正的结束token

        def parse_box_content(text: str):
            match = pattern.search(text)
            if match:
                content = match.group(1).strip()
                if content:
                    try:
                        content = torch.tensor(ast.literal_eval(content), dtype=input_ids.dtype, device=input_ids.device)
                        return content
                    except Exception:
                        return None
                return None
            return None

        ret = []
        for text in texts:
            res = parse_box_content(text)
            if res is not None:
                ret.append(res)
        return ret

    def _predict_mask_and_labels(self, input_ids, hidden_states, gt_masks, gt_labels, input_imgs):
        pred_masks = None
        pred_labels = None
        if self.add_cls_predict and gt_labels is not None:
            # 获取cls特征, 加入real / fake token的特征
            # cls_features = self.get_feature_by_interval(
            #     hidden_states,
            #     self.placeholder_ids[2],
            #     self.placeholder_ids[3],
            #     input_ids,
            # )
            # if cls_features is not None and cls_features.shape[1] == 1:
            #     # 还加入了|<cls>|的特征
            #     cls_pos_feat = self.get_embeddings_feature(hidden_states, self.detection_ids, input_ids)
            #     cls_features = torch.cat([cls_features, cls_pos_feat], dim=1)
            # 只选择1个token作为cls特征
            cls_features = self.get_embeddings_feature(hidden_states, self.detection_ids, input_ids)
            if cls_features is not None:
                pred_labels = self.cls_head(cls_features)

        if self.add_mask_predict and gt_masks is not None:
            mask_features = self.get_embeddings_feature(hidden_states, self.mask_ids, input_ids)
            # pos_features = self.get_embeddings_feature(hidden_states, self.position_ids, input_ids)
            # TODO: 仅使用mask_features和pos_features
            pos_features = None
            # 实现自动混合精度训练
            data_type = next(self.mask_head.model.parameters()).dtype
            box = self.get_bbox_text_content(input_ids)
            with torch.autocast(device_type=self.mask_head.device.type, dtype=data_type):
                pred_masks = self.mask_head(input_imgs, mask_features, pos_features, box)[0]
        return pred_masks, pred_labels

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()

    def forward(
        self,
        gt_masks: torch.LongTensor = None,
        gt_labels: torch.LongTensor = None,
        input_images: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        expert_embeds: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple, AlignLLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        """
        这段代码是Qwen2.5-VL(视觉语言)模型中将图像特征嵌入到模型输入的关键部分。
        它负责将视觉信息与文本信息融合在一起，使模型能够处理多模态输入。
        代码的主要作用是检查输入中是否有图像内容，如果有，则将图像特征与文本特征进行融合
        即将 图像特征 替换为 文本 embedding的内容
        """
        if inputs_embeds is None:
            # 转为embedding内容
            inputs_embeds = self.model.embed_tokens(input_ids)
            # 图像特征提取
            if image_embeds is None and pixel_values_videos is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            if n_image_tokens > 0:
                # 确保输入序列中的图像token数量与生成的图像特征数量一致。
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

                # 创建一个掩码，用于将图像特征插入到输入嵌入中
                mask = input_ids == self.config.image_token_id  # 查找图像token所在的位置
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)  # 扩展掩码以匹配输入嵌入的形状
                image_mask = mask_expanded.to(inputs_embeds.device)  # 将掩码转移到与输入嵌入相同的设备上

                # 融合特征
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # 将图像特征插入到输入嵌入中

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        """
        这段代码处理了Qwen2.5-VL模型中的位置编码(position IDs)计算，
        特别是关于RoPE（旋转位置编码）的实现。RoPE对于模型处理变长序列和多模态输入（文本、图像、视频）至关重要。
        """
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # TODO: 将input_ids_x改为通过输入参数传递
        input_ids = input_ids if input_ids is not None else kwargs.get("input_ids_x", None)

        # 如果是专家模型, 则需要获取专家模型的特征, 并将其嵌入
        if self.add_expert_feat:
            # 1. 构建专家token掩码（支持PyTorch 2.0+）
            expert_token_mask = torch.isin(input_ids, torch.tensor(self.expert_ids, device=self.device))
            # 2. 检查每个batch是否都包含专家token
            n_tokens_per_batch = expert_token_mask.sum(dim=1)
            if (n_tokens_per_batch == 0).any():
                # 只要有一个batch没有专家token，则不处理
                logger.debug("没有专家token, 不处理")
            else:
                if expert_embeds is None:
                    expert_embeds = self.get_expert_feat(input_images)
                batch_size, seq_len = input_ids.shape
                _, n_expert, hidden_dim = expert_embeds.shape
                # 只处理专家token数量和特征数量一致的batch
                valid_batch_mask = n_tokens_per_batch == n_expert
                if not valid_batch_mask.any():
                    logger.debug("所有batch专家token数量与特征数量不一致")
                else:
                    # 获取所有合法batch的索引
                    valid_indices = valid_batch_mask.nonzero(as_tuple=True)[0]
                    for b in valid_indices:
                        idx = expert_token_mask[b].nonzero(as_tuple=True)[0]
                        inputs_embeds[b, idx, :] = expert_embeds[b, : len(idx), :]

        # 大模型前向传播
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # TODO: 取每一层? 还是只取最后一层
        """
        last_hidden_state: 这是最后一层的隐藏状态，通常是一个形状为 [batch_size, sequence_length, hidden_size] 的张量，
        表示每个 token 在最后一层的语义表示。它通常用于后续任务（如分类、生成等）。

        output_attentions: 这是模型中各层注意力权重的集合，通常是一个 list 或 tuple，每个元素是一个张量，
        形状一般为 [batch_size, num_heads, sequence_length, sequence_length]。
        这些张量展示了每一层中各个注意力头对不同 token 的关注程度。
        
        attention_mask 是输入给模型的张量，用于告诉模型哪些 token 是有效的，哪些需要忽略（比如填充位置），从而在计算注意力得分时只考虑有效的 token。
        output_attentions 是一个布尔标记，当设置为 True 时，模型会将每一层（或部分层）的注意力权重也返回，方便进行调试或分析模型内部的注意力机制。
        """
        # 包含了每个 token 在最后一层的表示，后续会用于生成预测结果。
        hidden_states = outputs.last_hidden_state
        # 检查hidden_state和self.lm_head是否在同一个设备上
        if hidden_states.device != self.lm_head.weight.device:
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        logits = self.lm_head(hidden_states)
        pred_masks = None
        pred_labels = None
        loss = None
        loss_detail = {}

        pred_masks, pred_labels = self._predict_mask_and_labels(input_ids, hidden_states, gt_masks, gt_labels, input_images)
        if labels is not None:
            # 获取大模型的预测输出
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            # 计算损失
            loss, loss_detail = self.calculate_loss(pred_masks, pred_labels, shift_logits, gt_masks, gt_labels, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AlignLLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            pred_labels=pred_labels,
            gt_labels=gt_labels,
            **loss_detail,
        )

    def calculate_loss(self, pred_masks, pred_labels, pred_text, gt_masks, gt_labels, gt_text):
        """
        计算整体损失，同时返回各项损失详情。

        参数：
            pred_masks: 模型预测的mask；
            pred_labels: 模型预测的标签；
            pred_text: 模型预测的文本输出(logits)；
            gt_masks: 真实的mask；
            gt_labels: 真实的标签；
            gt_text: 真实的文本标签。

        返回：
            loss: 综合损失值；
            loss_detail: 详细的各项损失字典。
        """

        def safe_item(loss):
            return loss.item() if loss is not None else None

        # 计算文本损失
        loss_text = self.loss_text(pred_text, gt_text) * self.text_weight
        loss = loss_text
        loss_detail = {"loss_text": safe_item(loss_text)}

        # 如果没有标签预测，则只返回文本损失
        if pred_labels is None:
            return loss, loss_detail

        # 计算分类损失
        loss_cls = None
        if gt_labels is not None:
            loss_cls = self.loss_cls(pred_labels, gt_labels)
            loss += loss_cls
        loss_detail["loss_cls"] = safe_item(loss_cls)

        # 计算 mask 相关的损失
        loss_bce, loss_dice = None, None
        if gt_masks is not None and pred_masks is not None:
            loss_bce = self.loss_bce(pred_masks, gt_masks) * self.bce_weight
            if loss_bce < 0:
                logger.warning(f"loss_bce is negative: {loss_bce}, gt_labels: {gt_labels}")
            loss_dice = self.loss_dice(pred_masks, gt_masks)
            if loss_dice < 0:
                logger.warning(f"loss_dice is negative: {loss_dice}, gt_labels: {gt_labels}")
            loss_bce = torch.clamp(loss_bce, min=0.0)
            loss_dice = torch.clamp(loss_dice, min=0.0)
            loss += loss_bce + loss_dice
        loss_detail["loss_bce"] = safe_item(loss_bce)
        loss_detail["loss_dice"] = safe_item(loss_dice)

        return loss, loss_detail

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        kwargs.pop("input_ids_x", None)
        gt_labels = kwargs.pop("gt_labels", None)
        gt_imgs = kwargs.pop("gt_imgs", None)  # 专家模型需要用到
        gt_masks = kwargs.pop("gt_masks", None)
        pixel_values = kwargs["pixel_values"]
        image_grid_thw = kwargs["image_grid_thw"]
        kwargs["image_embeds"] = self.visual(pixel_values, grid_thw=image_grid_thw)
        if self.add_expert_feat:
            kwargs["expert_embeds"] = self.get_expert_feat(gt_imgs)

        # 设置生成的相应参数
        kwargs.update(
            {
                "do_sample": False,  # 禁用采样
                "temperature": 0.1,  # 保证生成的token是确定性的
                "top_p": 0.95,  # 保留概率最高的前95%的token
                "top_k": 10,  # 保留概率最高的前10个token
                "num_beams": 2,  # 轻量级beam search
                "no_repeat_ngram_size": 2,  # 避免n-gram重复
                "early_stopping": True,  # 提前停止
            }
        )
        sequences = super().generate(*args, **kwargs)
        if self.add_cls_predict is False and self.add_expert_feat is False:
            return sequences

        # 获取需要强制生成的 token，并将其拼接到生成结果序列末尾W
        generate_tokens = sequences.sequences[:, 1:]
        inputs = {
            "input_ids": generate_tokens,  # 去掉第一个占位符token
            "image_embeds": kwargs["image_embeds"],
            "image_grid_thw": image_grid_thw,
            "pixel_values": pixel_values,
            "gt_imgs": gt_imgs,
            "gt_masks": gt_masks,
            "gt_labels": gt_labels,
        }
        outputs = self.forward(**inputs, return_dict=True)
        # 将target_output的输出结果进行sigmoid处理
        target_output = {
            "pred_masks": outputs.pred_masks,
            "pred_labels": outputs.pred_labels,
        }
        # # 获取last_hidden_state
        return sequences | target_output

    def get_special_token(self):
        ret = []
        if self.add_cls_predict:
            ret += self.detection_ids
        if self.add_mask_predict:
            ret += self.mask_ids
        if self.add_expert_feat:
            ret += self.expert_ids
        return ret


def get_model_tokenizer_qwen_forensic(*args, **kwargs):
    kwargs["automodel_class"] = kwargs["automodel_class"] or QwenForensicModel
    return get_model_tokenizer_qwen2_vl(*args, **kwargs)


register_model(
    ModelMeta(
        ForensicModelType.qwen_model,
        [
            ModelGroup(
                [
                    Model("qwen_forensic_3B", "Qwen/Qwen2.5-VL-3B-Instruct"),
                    Model("qwen_forensic_7B", "Qwen/Qwen2.5-VL-7B-Instruct"),
                ]
            )
        ],
        TemplateType.qwen2_5_vl,
        get_function=get_model_tokenizer_qwen_forensic,
        model_arch="qwen2_vl",
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        tags=["vision"],
        is_multimodal=True,
    ),
)
