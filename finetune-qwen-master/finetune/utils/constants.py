from transformers.trainer_pt_utils import LabelSmoother

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE_INDEX = -100

DEFAULT_IMAGE_TOKEN = "<|img|>"
DEFAULT_REAL_IMAGE_TOKEN = "<|real_img|>"
DEFAULT_FAKE_IMAGE_TOKEN = "<|fake_img|>"

DEFAULT_EXPERT_TOKEN = "<|expert_{i}|>"
DEFAULT_MASK_TOKEN = "<|mask_{i}|>"
DEFAULT_CLASS_TOKEN = "<|cls|>"
DEFAULT_POS_TOKEN = "<|pos|>"

DEFAULT_RES_START_TOKEN = "<|res_start|>"
DEFAULT_RES_END_TOKEN = "<|res_end|>"
DEFAULT_BOX_START_TOKEN = "<|box_start|>"
DEFAULT_BOX_END_TOKEN = "<|box_end|>"


class ForensicTemplateType:
    forensic_template = "forensic_template"
    contrastive_template = "contrastive_template"
    controllable_template = "controllable_template"


class ForensicModelType:
    qwen_model = "qwen_forensic"
