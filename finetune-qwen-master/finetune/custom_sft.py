import time
from dataclasses import fields
from pathlib import Path
from typing import List, Union

from loguru import logger
from swift.llm.model.register import get_model_info_meta
from swift.llm.train import SwiftSft
from swift.trainers import IntervalStrategy
from swift.utils import get_model_parameter_info, patch_getattr
from transformers import AutoProcessor, PreTrainedTokenizerBase

from .arguments.custom_arguments import CustomTrainArguments, NewTrainArguments
from .datasets import load_data
from .models import QwenForensicModel
from .plugin.callback import extra_callbacks
from .trainer import CustomSeqTrainer
from .utils import LoggerConfig
from .templates import ForensicTemplate, ControllableTemplate, ContrastiveTemplate


class CustomSFT(SwiftSft):
    args_class = CustomTrainArguments
    args: args_class

    def __init__(self, args: Union[List[str], CustomTrainArguments, None] = None):
        # 实例化父类
        super().__init__(args)
        # 添加回调模型的代码内容
        self.callbacks += extra_callbacks

        # 由于output_dir的结构如下Qwen2.5-VL-3B-Instruct_sam2-hiera-tiny_3box_cls_mask/v0-20250430-095319
        LoggerConfig.configure(log_dir=self.args.output_dir, time_stamp="".join(self.args.output_dir.split("-")[-2:]))
        logger.info(f"参数配置清单如下:\n {self.args}")
        logger.info("====" * 20)

    def _prepare_model_tokenizer(self):
        args = self.args
        # 载入模型和processor
        kwargs = args.get_model_kwargs()
        # 从NewTrainArguments中获取模型
        model_kwargs = {}
        field_names = [f.name for f in fields(NewTrainArguments)]
        for name in field_names:
            if hasattr(args, name):
                model_kwargs[name] = getattr(args, name)

        self.train_dataset_name = []
        self.model: QwenForensicModel = QwenForensicModel.from_pretrained(
            kwargs["model_id_or_path"],
            torch_dtype=kwargs["torch_dtype"],
            attn_implementation="flash_attention_2" if "flash" in kwargs["attn_impl"] else kwargs["attn_impl"],
            use_cache=True,
            local_files_only=True,
            **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(kwargs["model_id_or_path"], use_fast=True)
        model_info, model_meta = get_model_info_meta(**kwargs)

        # 打补丁, 虽然没什么用
        if not isinstance(self.processor, PreTrainedTokenizerBase) and hasattr(self.processor, "tokenizer"):
            tokenizer = self.processor.tokenizer
            patch_getattr(self.processor.__class__, "tokenizer")
        else:
            tokenizer = self.processor

        tokenizer.model_info = model_info
        tokenizer.model_meta = model_meta
        self.model.setup_tokens_for_conversation(tokenizer, self.args.sam_model_path)

        self.model.model_info = model_info
        self.model.model_meta = model_meta
        self._prepare_generation_config()
        self._prepare_gradient_checkpointing()

    def get_data_module(self):
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()

        train_dataset_list = []
        for path in args.dataset:
            target_path = Path(path)
            self.train_dataset_name.append(target_path.name)
            assert target_path.exists(), "Target Train Dataset is not exists"
            train_dataset_list.append(
                {
                    "dataset_path": target_path.absolute().as_posix(),
                    "pred_mask": self.args.add_mask_predict,
                    "resize_image": 1024,
                    "add_expert_feat": self.args.add_expert_feat,
                    "split_box_num": self.args.split_box_num,
                    "token_pos": self.args.token_pos,
                    "conversation_templates": self.args.conversation_templates,
                }
            )

        train_dataset, val_dataset = load_data(
            train_dataset_list,
            split_dataset_ratio=args.split_dataset_ratio,
            template=self.template,
            dataset_type=self.args.dataset_type,
            **dataset_kwargs,
        )

        if len(args.val_dataset) > 0:
            # Loading val dataset
            assert args.split_dataset_ratio == 0.0, "The val_dataset should not be split from the train_dataset"
            val_dataset_list = []
            for path in args.val_dataset:
                val_dataset_list.append(
                    {
                        "dataset_path": Path(path).absolute().as_posix(),
                        "pred_mask": True,
                        "resize_image": 1024,
                        "split_box_num": self.args.split_box_num,
                        "conversation_templates": self.args.conversation_templates,
                    }
                )
            _, val_dataset = load_data(
                val_dataset_list,
                template=self.template,
                split_dataset_ratio=1.0,
                dataset_type=self.args.dataset_type,
                **dataset_kwargs,
            )

        logger.info(f"The length of train_dataset : {len(train_dataset)}")
        if val_dataset is not None:
            logger.info(f"The length of eval_dataset: {len(val_dataset)}")
        else:
            logger.info(f"No eval_dataset is provided")

        if val_dataset is None:
            args.training_args.evaluation_strategy = IntervalStrategy.NO
            args.training_args.eval_strategy = IntervalStrategy.NO

        # 加载数据集聚合器
        data_collator = self._get_data_collator()

        data_module = dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)
        trainer_kwargs: dict = self._get_trainer_kwargs()

        # 将data_module与trainer_kwargs合并, 重复的key以data_module为准
        return trainer_kwargs | data_module

    def run(self):
        # 载入数据集与配置条目
        data_module = self.get_data_module()

        # 获取微调模型
        self.model = self.prepare_model(
            self.args,
            self.model,
            template=self.template,
            train_dataset=data_module["train_dataset"],
        )
        logger.info(f"model: {self.model}")

        # 获取微调参数
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg["model_parameter_info"] = model_parameter_info
        logger.info(f"model_parameter_info: {model_parameter_info}")

        # 训练, 指定自定义训练器
        # trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer_cls = CustomSeqTrainer
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            callbacks=self.callbacks,
            template=self.template,
            **data_module,
        )
        return self.train(trainer)
