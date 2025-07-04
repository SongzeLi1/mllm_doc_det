# from torch.nn import LayerNorm
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from swift.llm.infer.protocol import ChatCompletionResponse
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)


@dataclass
class NewField(OrderedDict):
    # 新增字段
    gt_masks: Optional[torch.FloatTensor] = None
    pred_masks: Optional[torch.FloatTensor] = None
    gt_labels: Optional[torch.FloatTensor] = None
    pred_labels: Optional[torch.FloatTensor] = None
    loss_cls: Optional[torch.FloatTensor] = None
    loss_bce: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_text: Optional[torch.FloatTensor] = None


@dataclass
class AlignLLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast, NewField):
    pass


@dataclass
class ForensicChatCompletionResponse(ChatCompletionResponse):
    pred_masks: Optional[torch.FloatTensor] = None
    pred_labels: Optional[torch.FloatTensor] = None
