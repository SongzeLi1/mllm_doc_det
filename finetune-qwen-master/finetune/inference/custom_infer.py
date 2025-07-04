import datetime
import json
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from PIL import Image
from swift.llm import SwiftPipeline, merge_lora
from swift.llm.infer import SwiftInfer
from swift.llm.infer.infer_engine import AdapterRequest
from swift.plugin import InferStats

from ..arguments.custom_arguments import CustomInferArguments
from ..datasets import load_data
from ..models import QwenForensicModel
from ..models.utils import prepare_model_template
from ..utils.json_decoder import JsonDecoder
from ..utils.logger import LoggerConfig
from . import InferRequestLoader
from .custom_engine import CustomEngine


# 在文件顶部或合适位置添加辅助函数
def default_encoder(o):
    try:
        return o.tolist()
    except AttributeError:
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


class CustomInfer(SwiftInfer):
    """推理引擎封装类，提供统一的推理接口"""

    args_class = CustomInferArguments

    def __init__(self, args: Union[List[str], CustomInferArguments, None] = None) -> None:
        # 实例化祖父类
        super(SwiftInfer, self).__init__(args)
        args = self.args
        # 获取当前时间戳
        time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")[:-1]
        save_name = Path(args.adapters[0]).name
        if os.getenv("IS_DEBUG") == "1":
            save_dir = Path(os.getcwd()) / "test_output" / Path(args.adapters[0]).parent.parent.name
        else:
            save_dir = Path(args.result_path).parent.parent.parent / Path(args.adapters[0]).parent.parent.name
        args.result_path = save_dir / time_stamp
        LoggerConfig.configure(save_dir, time_stamp, logger_name=save_name)

        if args.merge_lora:
            merge_lora(args, device_map="cpu")
        self.backend = args.infer_backend
        self.infer_kwargs = {}
        if self.backend == "vllm" and args.adapters:
            self.infer_kwargs["adapter_request"] = AdapterRequest("_lora", args.adapters[0])
        elif self.backend == "pt":
            self.model, self.template = prepare_model_template(
                args,
                automodel_class=QwenForensicModel,
                model_kwargs=args.get_custom_arguments(),
            )
            self.infer_engine = CustomEngine.from_model_template(
                self.model,
                self.template,
                max_batch_size=args.max_batch_size,
            )
            logger.info(f"model: {self.infer_engine.model}")
        else:
            self.infer_engine = self.get_infer_engine(args)
            self.template = args.get_template(self.processor)
        self.random_state = np.random.RandomState(args.data_seed)

    def aggregate_results(self, result_list: dict) -> dict:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_results = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(all_results, result_list)
            merged = {}
            for r in all_results:
                merged.update(r)
            return merged
        return result_list

    def infer_dataset(self):
        args = self.args
        request_config = args.get_request_config()
        logger.info(f"request_config: {request_config}")
        self.infer_kwargs["metrics"] = [InferStats()]
        if request_config and request_config.stream:
            raise NotImplementedError("Stream mode is not supported yet.")
        else:
            all_result_list = {}
            # 针对每个数据集分别推理和保存
            # 读取val_dataset
            with open(args.val_dataset[0], "r", encoding="utf-8") as f:
                val_dataset_list = json.load(f)

            for val_dataset_name, val_dataset_path in val_dataset_list.items():
                torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                logger.info(f"正在推理数据集: {val_dataset_path}")
                # 只处理当前数据集
                single_val_dataset_list = [
                    {
                        "dataset_path": Path(val_dataset_path).absolute().as_posix(),
                        "pred_mask": args.add_mask_predict,
                        "resize_image": 1024,
                        "add_expert_feat": args.add_expert_feat,
                        "token_pos": args.token_pos,
                    }
                ]
                _, val_dataset = load_data(
                    single_val_dataset_list,
                    self.template,
                    split_dataset_ratio=1.0,
                    seed=self.random_state,
                )
                if val_dataset is None or len(val_dataset) == 0:
                    logger.warning(f"Validation dataset {val_dataset_path} is empty, skip.")
                    continue

                # 判断是否分布式训练
                if dist.is_initialized():
                    val_dataset = InferRequestLoader.distribute(val_dataset, batch_size=args.max_batch_size)
                else:
                    val_dataset = InferRequestLoader.distribute(
                        val_dataset, batch_size=args.max_batch_size, distributed=False
                    )

                # 针对每个数据集单独保存

                resp_list = self.infer_engine.infer(
                    val_dataset,
                    request_config,
                    template=self.template,
                    use_tqdm=True,
                    save_path=args.result_path / val_dataset_name,
                    **self.infer_kwargs,
                )

                # 聚合多卡结果
                result_list = self.aggregate_results(resp_list)
                json_save_path: Path = args.result_path / f"{val_dataset_name}.json"
                if self.args.rank == 0 or dist.is_initialized() is False:
                    # 保存jsonl
                    with open(json_save_path, "w", encoding="utf-8") as f:
                        json.dump(result_list, f, ensure_ascii=False, indent=4)
                    logger.info(f"推理结果已保存到: {json_save_path}")
                    if args.metric is not None:
                        self._calc_metric()
                all_result_list[val_dataset_name] = result_list
        return all_result_list
