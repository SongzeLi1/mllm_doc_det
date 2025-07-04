import traceback
from typing import List, Union

from loguru import logger

from finetune.arguments.custom_arguments import CustomInferArguments
from finetune.inference.custom_infer import CustomInfer

"""
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048
"""


def infer_main(args: Union[List[str], CustomInferArguments, None] = None):
    try:
        custom_inference = CustomInfer(args)
        custom_inference.main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.debug(traceback.format_exc())
        raise e


if __name__ == "__main__":
    infer_main()
