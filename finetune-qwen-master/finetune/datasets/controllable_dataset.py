from collections import OrderedDict
import json
from loguru import logger
from PIL import Image, ImageFile
import numpy as np
from .custom_dataset import CustomDataset
from ..utils.constants import DEFAULT_BOX_END_TOKEN, DEFAULT_BOX_START_TOKEN
from ..utils.image_similarity import ImageSimilarity

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ControllableDataset(CustomDataset):
    def __init__(self, gt_type="lpips", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_similarity = ImageSimilarity()
        self.gt_type = gt_type

    def data_processing(self, index):
        info_data = self.data[index]
        real_img_path = info_data["real_image"]
        fake_img_path = info_data["fake_image"]
        instruct = info_data["forensic_analysis"]

        # 处理图片
        real_image = self.load_image(real_img_path)
        fake_image = self.load_image(fake_img_path)

        # 重设图片大小
        fake_image = self.image_similarity.check_and_resize(real_image, fake_image)

        if self.gt_type == "lpips":
            gt_mask = self.image_similarity.compare_images_lpips(real_image, fake_image)
        else:
            gt_mask = self.image_similarity.compare_images_pixel(real_image, fake_image)

        ret = self.common_transform(image=np.array(fake_image), mask=np.array(gt_mask))
        pos_ret = self.post_transform(image=ret["image"], mask=ret["mask"])
        gt_label = 1
        np_img, np_mask = pos_ret["image"], pos_ret.get("mask", None)

        return fake_img_path, np_img, np_mask, gt_label, instruct

    def __getitem__(self, idx):
        # 选择模板
        try:
            fake_img_path, fake_np_image, gt_np_img, gt_label, instruct = self.data_processing(idx)
            user_msg, assist_msg, gt_boxes = self.get_conversation(gt_np_img, gt_label, instruct)
            # 生成数据
            data = {
                "images": [{"bytes": Image.fromarray(fake_np_image)}],
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assist_msg},
                ],
                "image": fake_np_image,
                "gt_mask": gt_np_img,
                "gt_label": gt_label,
                "gt_boxes": gt_boxes,
            }
            if self.encode_func is None:
                return data
            return self.encode_func.encode(data, return_template_inputs=True)
        except Exception as e:
            logger.error(f"Error in get_conversation: {e}. fake_img_path: {fake_img_path}")
            raise e
