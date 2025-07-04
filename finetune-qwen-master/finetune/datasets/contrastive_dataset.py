import json
from loguru import logger
from PIL import Image, ImageFile
from .custom_dataset import CustomDataset
from ..utils.constants import DEFAULT_BOX_END_TOKEN, DEFAULT_BOX_START_TOKEN

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ContrastiveDataset(CustomDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def data_processing(self, index):
        info_data = self.data[index]
        real_img_path = info_data["real_image"]
        fake_img_path = info_data["fake_image"]
        gt_img_path = info_data["gt_image"]
        instruct = info_data["diff"]

        # 处理图片
        real_image = self.load_image(real_img_path)
        fake_image = self.load_image(fake_img_path)
        gt_img = self.load_image(gt_img_path)
        return real_img_path, fake_img_path, gt_img_path, real_image, fake_image, gt_img, instruct

    def get_conversation(self, gt_mask, instruct, result_separator="<res>"):
        conversation_template = self.random_state.choice(self.conversation_templates)
        assistant_message: str = conversation_template["assistant"]
        assert result_separator in assistant_message, f"Assistant message not contain result_separator: {result_separator}"

        # TODO: 要求预测 [图像描述 & 差异描述 & mask的box级别差异]
        pos_info, positions = self.get_image_pos_info(gt_mask, label=None, is_ploygons=False, is_bbox=True)
        instruct["mask"] = DEFAULT_BOX_START_TOKEN + pos_info + DEFAULT_BOX_END_TOKEN
        # instruct是一个dict, 转为json str
        if isinstance(instruct, dict):
            instruct = json.dumps(instruct, ensure_ascii=False)
        elif not isinstance(instruct, str):
            raise ValueError(f"instruct should be a dict or str, but got {type(instruct)}")
        assistant_message = assistant_message.replace(result_separator, instruct)
        user_message = conversation_template["user"]
        return user_message, assistant_message, positions

    def __getitem__(self, idx):
        # 选择模板
        try:
            real_img_path, fake_img_path, gt_img_path, real_image, fake_image, gt_mask, instruct = self.data_processing(idx)
            user_message, assistant_message, gt_boxes = self.get_conversation(gt_mask, instruct)
            # np 转为 PIL
            # 生成数据
            data = {
                "images": [{"bytes": real_image}, {"bytes": fake_image}],
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ],
                "real_img_path": real_img_path,
                "fake_img_path": fake_img_path,
                "gt_img_path": gt_img_path,
                "gt_masks": gt_mask,
                "gt_boxes": gt_boxes,
            }
            if self.encode_func is None:
                return data
            return self.encode_func.encode(data, return_template_inputs=True)
        except Exception as e:
            logger.error(f"Error in get_conversation: {e}. real_img_path: {real_img_path}, fake_img_path: {fake_img_path}")
            raise e
