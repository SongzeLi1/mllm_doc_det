import os
import sys
from pathlib import Path

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


mask_path = Path("/pubdata/yuyangxin/swift-demo/resource/mask.png")


mask = Image.open(mask_path)

from finetune.utils.polygon_convert import mask_to_polygons, polygons_to_mask

result = mask_to_polygons(mask)
print(result)
