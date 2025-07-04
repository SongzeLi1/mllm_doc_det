from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from swift.llm import InferRequest
from swift.llm.template import register_template
from swift.llm.template.template.qwen import Qwen2_5VLTemplate, QwenTemplateMeta
from swift.llm.template.template_inputs import TemplateInputs

from ..utils.constants import ForensicTemplateType


@dataclass
class ContrastiveTemplate(Qwen2_5VLTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.truncation_strategy = "right"

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        将输入的batch数据进行处理，返回处理后的数据
        """
        collator = super()._data_collator(batch, padding_to=padding_to)

        # # 堆叠gt_labels, gt_masks成tensor
        if "gt_img" in batch[0]:
            collator["gt_imgs"] = [x["gt_img"] for x in batch]
        if "gt_label" in batch[0]:
            collator["gt_labels"] = torch.tensor(np.stack([x["gt_label"] for x in batch]), dtype=collator["pixel_values"].dtype)
        if batch[0].get("gt_mask", None) is not None:
            try:
                collator["gt_masks"] = torch.tensor(np.stack([x["gt_mask"] for x in batch]), dtype=collator["pixel_values"].dtype)
            except Exception as e:
                collator["gt_masks"] = None
        return collator

    def _post_encode(self, model, inputs):
        """
        将图像使用visual模型进行编码, 然后嵌入到embeds当中
        """
        inputs["pixel_values"] = inputs["pixel_values"].requires_grad_(True)

        # 执行后处理编码
        post_res = super()._post_encode(model, inputs)
        post_res["input_ids_x"] = inputs["input_ids"]
        if "gt_imgs" in inputs:
            post_res["gt_imgs"] = inputs["gt_imgs"]
        if "gt_labels" in inputs:
            post_res["gt_labels"] = inputs["gt_labels"]
        if "gt_masks" in inputs:
            post_res["gt_masks"] = inputs["gt_masks"]
        return post_res

    @torch.inference_mode()
    def encode(
        self,
        inputs: Union[TemplateInputs, Dict[str, Any], InferRequest],
        return_template_inputs: bool = False,
    ) -> Dict[str, Any]:
        """
        @torch.inference_mode() 是 PyTorch 中的一个装饰器，应用在 QwenForensicTemplate 类的 encode 方法上，它具有以下作用：
        禁用梯度计算：它会关闭 PyTorch 的自动微分功能，所有在该方法中执行的张量操作不会被记录用于反向传播。
        优化推理性能：减少内存使用（不存储计算图）、提高运行速度（跳过梯度相关的计算开销）、优化 CUDA 相关操作
        比 torch.no_grad() 更严格：它提供了比 torch.no_grad() 更严格的优化，完全省略了与梯度相关的所有内部机制。
        """
        # 将不存在encoded中的字段从src_inputs中加入到encoded中
        src_inputs = inputs
        encoded = super().encode(inputs, return_template_inputs=return_template_inputs)
        if isinstance(src_inputs, dict):
            for key in src_inputs:
                if key not in encoded:
                    encoded[key] = src_inputs[key]
        return encoded


register_template(QwenTemplateMeta(ForensicTemplateType.contrastive_template, template_cls=ContrastiveTemplate))
