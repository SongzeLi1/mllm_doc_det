import json
from pathlib import Path

import numpy as np
from loguru import logger
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class EarlyStopCallback(TrainerCallback):
    """An early stop implementation"""

    def __init__(self, total_interval=5):
        self.best_metric = None
        self.interval = 0
        self.total_interval = total_interval

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 保存模型的config配置参数
        # 将args保存到为json文件
        output_dir = args.output_dir
        args_save_path = Path(output_dir) / f"training_args_steps{state.global_step}.json"
        args_save_path.write_text(args.to_json_string(), encoding="utf-8")

        # 早停代码逻辑
        operator = np.greater if args.greater_is_better else np.less
        if self.best_metric is None or operator(state.best_metric, self.best_metric):
            self.best_metric = state.best_metric
        else:
            self.interval += 1

        if self.interval >= self.total_interval:
            logger.info(f"Training stop because of eval metric is stable at step {state.global_step}")
            control.should_training_stop = True


extra_callbacks = [EarlyStopCallback()]
