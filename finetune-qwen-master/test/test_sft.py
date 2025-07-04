import sys

# 添加父目录到 Python 路径
sys.path.append("/pubdata/yuyangxin/swift-demo")

from typing import List, Union

from swift.llm import TrainArguments
from swift.llm.argument import TrainArguments

from finetune import CustomSFT


def pt_main(args: Union[List[str], TrainArguments, None] = None):
    return CustomSFT(args).main()


if __name__ == "__main__":
    pt_main()
