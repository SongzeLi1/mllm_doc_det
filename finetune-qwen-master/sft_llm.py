import traceback
from typing import List, Union

from loguru import logger

from finetune.arguments import CustomTrainArguments
from finetune.custom_sft import CustomSFT


def main(args: Union[List[str], CustomTrainArguments, None] = None):
    try:
        CustomSFT(args).main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.debug(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
