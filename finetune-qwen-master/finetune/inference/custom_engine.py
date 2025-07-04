import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from PIL import Image
from swift.llm import InferRequest, Template
from swift.llm.infer.infer_engine import AdapterRequest, PtEngine
from swift.llm.infer.infer_engine.utils import AdapterRequest
from swift.llm.infer.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    RequestConfig,
)
from swift.plugin import Metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig

from ..utils.logger import ORIG_STDOUT


class CustomEngine(PtEngine):

    def _infer_full(
        self,
        template: Template,
        inputs: Dict[str, Any],
        *,
        generation_config: GenerationConfig,
        adapter_request: Optional[AdapterRequest] = None,
        template_inputs=None,
    ) -> List[ChatCompletionResponse]:
        # bos_token TODO: encoder-decoder
        generate_kwargs = {"generation_config": generation_config, **inputs}
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            generate_kwargs["adapter_names"] = adapter_names
        num_prompt_tokens = self._get_num_tokens(inputs)
        generate_kwargs = template.prepare_generate_kwargs(generate_kwargs, model=self.model)
        output = dict(template.generate(self.model, **generate_kwargs))
        output.pop("past_key_values", None)
        batched_generate_ids = output["sequences"]
        batched_generate_ids = template.get_generate_ids(batched_generate_ids, num_prompt_tokens)
        template.debug_logger({"generate_ids": batched_generate_ids})  # debug
        batched_logprobs = self.preprocess_logits(
            output.get("logits"), batched_generate_ids, generation_config.top_logprobs
        )

        res = []
        num_return_sequences = generation_config.num_return_sequences
        for i in range(inputs["attention_mask"].shape[0]):
            choices = []
            usage_info = self._get_usage_info(num_prompt_tokens, 0)
            for j in range(num_return_sequences):
                batched_index = i * num_return_sequences + j
                generate_ids = batched_generate_ids[batched_index]

                # ignore pad_token
                masks = generate_ids != self.tokenizer.pad_token_id
                generate_ids = generate_ids[masks].tolist()
                logprobs_list = None
                if batched_logprobs is not None:
                    logprobs_list = [
                        logprobs for m, logprobs in zip(masks, batched_logprobs[batched_index]) if m.item()
                    ]

                logprobs = self._get_logprobs(logprobs_list, generate_ids, generation_config.top_logprobs)
                usage_info = self._update_usage_info(usage_info, len(generate_ids))
                # response = template.decode(generate_ids, template_inputs=template_inputs[i])
                # 不解析特殊token
                response = template.decode(
                    generate_ids,
                    template_inputs=template_inputs[i],
                    tokenizer_kwargs={"skip_special_tokens": True},
                )

                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, num_prompt_tokens, True)
                tool_call = self._get_toolcall(response, template)
                choices.append(
                    ChatCompletionResponseChoice(
                        index=j,
                        message=ChatMessage(role="assistant", content=response, tool_calls=tool_call),
                        finish_reason=finish_reason,
                        logprobs=logprobs,
                    )
                )
            res.append(ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info))
        return {
            "result": res,
            "pred_masks": (
                torch.sigmoid(output.get("pred_masks").to(torch.float32)).cpu().numpy()
                if output.get("pred_masks") is not None
                else [None] * len(res)
            ),
            "pred_labels": (
                torch.sigmoid(output.get("pred_labels").to(torch.float32)).cpu().numpy().tolist()
                if output.get("pred_labels") is not None
                else [None] * len(res)
            ),
        }

    def infer(
        self,
        infer_requests: Union[List[InferRequest], DataLoader],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
        save_path: Optional[Path] = None,
    ):
        # 判断当前是否是主进程：当没有分布式环境或当前 rank 为 0 时视为主进程
        is_master = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

        if request_config is None:
            request_config = RequestConfig()
        if request_config.stream:
            return super().infer(
                infer_requests,
                request_config,
                metrics,
                template=template,
                use_tqdm=use_tqdm,
                adapter_request=adapter_request,
            )

        logger.info(f"Start load with {len(infer_requests)} samples. Batch size: {self.max_batch_size}")
        if isinstance(infer_requests, DataLoader):
            batched_iter = infer_requests  # DataLoader yields List[InferRequest] already
        else:
            max_batch_size = self.max_batch_size or (
                len(infer_requests) if not isinstance(infer_requests, DataLoader) else None
            )
            batched_iter = (
                infer_requests[i : i + max_batch_size] for i in range(0, len(infer_requests), max_batch_size)
            )

        # 仅当是主进程时启用进度条
        if use_tqdm is None:
            use_tqdm = is_master and (not request_config.stream and len(infer_requests) > 1)

        logger.info(f"开始执行推理...")
        if is_master:
            if isinstance(infer_requests, DataLoader) and dist.is_initialized():
                # 多卡环境下，将全量样本数平分
                world_size = dist.get_world_size()
                bar_len = math.ceil(len(infer_requests.dataset) / world_size)
            elif isinstance(infer_requests, DataLoader):
                bar_len = len(infer_requests.dataset)
            else:
                bar_len = len(infer_requests)
            prog_bar = tqdm(total=bar_len, dynamic_ncols=True, file=ORIG_STDOUT, disable=not use_tqdm)
        else:
            prog_bar = None

        ret = {}
        for infer_requests_samples in batched_iter:
            # 执行推理
            infer_res = self._infer(
                infer_requests_samples, request_config, template=template, adapter_request=adapter_request
            )
            infer_res["result"] = self._update_metrics(infer_res["result"], metrics)
            # 更新策略
            for info, pred_mask, pred_label, result in zip(
                infer_requests_samples,
                infer_res["pred_masks"],
                infer_res["pred_labels"],
                infer_res["result"],
            ):
                gt_mask_save_path, pred_mask_save_path = self.save_image(
                    save_path, info["image_path"], pred_mask, info["gt_mask"]
                )
                ret[info["image_path"]] = {
                    "gt_mask_path": gt_mask_save_path,
                    "pred_mask_path": pred_mask_save_path,
                    "gt_label": info["gt_label"],
                    "pred_label": pred_label,
                    "content": result.choices[0].message.content,
                }
            if is_master and prog_bar is not None:
                # 更新进度条
                prog_bar.update(len(infer_requests_samples))
        if is_master and prog_bar is not None:
            prog_bar.close()
        return ret

    def save_image(self, save_path, image_path, pred_mask_np, gt_mask_np):
        # 针对每个数据集单独保存
        pred_masks_save_dir = save_path / "pred_masks"
        gt_masks_save_dir = save_path / "gt_masks"
        pred_masks_save_dir.mkdir(parents=True, exist_ok=True)
        gt_masks_save_dir.mkdir(parents=True, exist_ok=True)

        # 保存mask图片
        if pred_mask_np is not None and gt_mask_np is not None:
            pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
            pred_mask_np = np.squeeze(pred_mask_np)
            pred_mask_save_path = (pred_masks_save_dir / Path(image_path).name).as_posix()
            Image.fromarray(pred_mask_np).save(pred_mask_save_path)

            gt_mask_np = (gt_mask_np * 255).astype(np.uint8)
            gt_mask_np = np.squeeze(gt_mask_np)
            gt_mask_save_path = (gt_masks_save_dir / Path(image_path).name).as_posix()
            Image.fromarray(gt_mask_np).save(gt_mask_save_path)
        else:
            gt_mask_save_path = None
            pred_mask_save_path = None
        return gt_mask_save_path, pred_mask_save_path
