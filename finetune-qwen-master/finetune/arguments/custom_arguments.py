from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from swift.llm import InferArguments
from swift.llm.argument.train_args import TrainArguments

from ..utils import LoggerConfig


@dataclass
class SAMConfigArguments:
    # SAM模型路径
    sam_model_path: str = field(
        default="facebook/sam2-hiera-tiny",
        metadata={"help": "Path to the SAM model checkpoint."},
    )
    # 添加split_box的参数截断
    split_box_num: int = field(
        default=3,
        metadata={"help": "box的分割数量"},
    )


@dataclass
class NewTrainArguments(SAMConfigArguments):
    # 添加token的位置信息
    token_pos: int = field(
        default=0,
        metadata={"help": "token的位置信息, 0:在所有之前; 1:在所有之后; 2:在回答之前; 在3:在回答之后"},
    )
    # 添加stage的训练截断
    train_stage: int = field(
        default=1,
        metadata={"help": "训练的3个阶段, 1:只训练LLM; 2. 只训练SAM2; 3. 训练LLM和SAM2"},
    )

    # 是否添加mask掩码损失
    add_mask_predict: bool = field(
        default=True,
        metadata={"help": "Whether to add mask loss."},
    )
    # 是否添加分类预测部分的内容
    add_cls_predict: bool = field(
        default=True,
        metadata={"help": "Whether to add predict model."},
    )
    # 是否添加专家模型
    add_expert_feat: bool = field(
        default=False,
        metadata={"help": "Whether to add expert model."},
    )

    # 模型输出配置
    num_new_tokens: int = field(
        default=32,
        metadata={"help": "Number of new tokens to generate."},
    )

    # 图像resize大小
    resize_image_size: int = field(
        default=1024,
        metadata={"help": "Size to resize the image."},
    )

    text_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for text loss."},
    )

    cls_weight: float = field(
        default=2.0,
        metadata={"help": "Weight for classification loss."},
    )

    bce_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for binary cross entropy loss."},
    )

    dice_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for dice loss."},
    )

    expert_model_weight_path: str = field(
        default="/data0/yuyangxin/finetune-qwen/resource/weight/nprmodel.pth",
        metadata={"help": "Path to the trufor config file."},
    )

    # custom, controllable, contrastive
    dataset_type: str = field(
        default="custom",
        metadata={
            "help": "Dataset type to use. Options: ['custom', 'controllable', 'contrastive']",
            "choices": ["custom", "controllable", "contrastive"],
        },
    )

    conversation_templates: str = field(
        default=None,
        metadata={"help": "Path to the conversation templates."},
    )

    def get_custom_arguments(self):
        new_train_fields = {f.name for f in fields(NewTrainArguments)}
        # 将子类实例转换为字典，并过滤掉不在 NewTrainArguments 中的键
        filtered_dict = {k: v for k, v in asdict(self).items() if k in new_train_fields}
        return filtered_dict


@dataclass
class CustomTrainArguments(NewTrainArguments, TrainArguments):

    def _init_output_dir(self):
        if self.output_dir is None:
            self.output_dir = f"output/{self.model_suffix}"
        sam_model_info = self.sam_model_path.split("/")
        if len(sam_model_info) == 2:
            sam_model_info = sam_model_info[1]
        else:
            sam_model_info = sam_model_info[-3]

        model_name = f"{self.model_suffix}_{sam_model_info}_box{self.split_box_num}_pos{self.token_pos}"
        if self.add_cls_predict:
            model_name += "_cls"
        if self.add_mask_predict:
            model_name += "_mask"
        if self.add_expert_feat:
            model_name += "_expert"
        if self.dataset_type != "custom":
            model_name += f"_{self.dataset_type}"

        self.output_dir: Path = Path(self.output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.output_dir.absolute().as_posix()


@dataclass
class CustomInferArguments(NewTrainArguments, InferArguments):
    pass
