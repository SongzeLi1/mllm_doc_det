from dataclasses import dataclass, field

from PIL import Image
from swift.llm.template.template_inputs import InferRequest


@dataclass
class CustomRequest(InferRequest):
    gt_label: int = field(default=None)  # 或其他合适的默认值
    gt_mask: Image.Image = field(default=None)
