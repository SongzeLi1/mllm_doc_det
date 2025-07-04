import concurrent.futures
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm  # 用于显示进度条

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def opt_dataset(json_path, data_dir):
    json_data = read_json(json_path)
    mask_dir = Path(data_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)  # 修复：将mask_path改为mask_dir
    image_dir = Path(data_dir) / "images"

    ret = []
    real, fake = 0, 0
    for i in json_data:
        for name, info in i.items():
            img_path = Path(image_dir) / info["img_file_name"]

            # 将多边形列表转换为多组顶点数组，转换为 int32 并 reshape(-1,2)
            polygons = []
            for ref in info["refs"]:
                for seg in ref["segmentation"]:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    polygons.append(pts)

            if not polygons:
                label = 0
                mask_path = "positive"
                real += 1
            else:
                label = 1
                image = cv2.imread(str(img_path))
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, polygons, 255)
                mask_path = mask_dir / info["img_file_name"]
                cv2.imwrite(str(mask_path), mask)
                fake += 1
            ret.append(
                (
                    str(img_path),
                    str(mask_path),
                    label,
                )
            )

    # 保存为 JSON 文件
    print(f"real: {real}, fake: {fake}")
    save_path = "/pubdata/yuyangxin/swift-demo/resource/datasets/without_instruct/synthscars_test.json"
    with open(save_path, "w") as f:
        json.dump(ret, f, indent=4, ensure_ascii=False)
    return ret


if __name__ == "__main__":
    json_path = "/pubdata/yuyangxin/dataset/synthscars/test/annotations/test.json"
    data_dir = "/pubdata/yuyangxin/dataset/synthscars/test"
    opt_dataset(json_path, data_dir)
