import json
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger
from PIL import Image, ImageFile
from swift.llm.template import Template

from ..utils import MaskPolygonConverter
from ..utils.constants import (
    DEFAULT_BOX_END_TOKEN,
    DEFAULT_BOX_START_TOKEN,
    DEFAULT_CLASS_TOKEN,
    DEFAULT_EXPERT_TOKEN,
    DEFAULT_MASK_TOKEN,
    DEFAULT_POS_TOKEN,
    DEFAULT_RES_END_TOKEN,
    DEFAULT_RES_START_TOKEN,
)
from .base import BaseDataset, TokenPosEnum
from .iml_transforms import get_albu_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDataset(BaseDataset):
    def __init__(
        self,
        dataset_path: str,
        pred_mask: bool,
        *,
        encode_func: Template = None,
        conversation_templates: str = None,
        num_new_tokens: int = 32,
        add_expert_feat: bool = False,
        resize_image=1024,
        random_state: Union[np.random.RandomState, int, None] = None,
        traceback_limit: int = 10,
        split_box_num: int = 3,
        token_pos: int = 0,
    ) -> None:
        self.pred_mask = pred_mask
        self.encode_func = encode_func
        self.dataset_path = Path(dataset_path)
        self.add_expert_feat = add_expert_feat
        self.split_box_num = split_box_num
        self.token_pos = TokenPosEnum(token_pos)
        logger.info(f"采用的特殊Token位置信息为: {self.token_pos}")
        assert self.dataset_path.exists(), f"Dataset path {dataset_path} does not exist."
        if conversation_templates is None:
            self.conversation_path = Path(__file__).parent.parent / "templates" / "forensic_template.txt"
        else:
            self.conversation_path = Path(conversation_templates)
        assert self.conversation_path.exists(), f"Conversation templates path {self.conversation_path} does not exist."

        # 对数据进行编码, 图像增广和处理机制
        # ANS: 经实验, 不pad能够提升性能
        self.common_transform = get_albu_transforms("pad", normalize=False, output_size=None)
        self.post_transform = get_albu_transforms("resize", normalize=False, output_size=(resize_image, resize_image))
        self.num_new_tokens = num_new_tokens

        with open(dataset_path, "r") as f:
            self.data = json.load(f)

        # TEMP: 临时断点重测的脚本内容
        # completed_data = JsonDecoder.load_file(
        #     "/pubdata/yuyangxin/swift-demo/result/Qwen2.5-VL-3B-Instruct/infer_result/20250423-212620.jsonl"
        # )
        # # 从self.data中去掉已经完成的数据
        # new_data = []
        # for item in self.data:
        #     if item[0] in completed_data:
        #         continue
        #     new_data.append(item)
        # self.data = new_data

        self.conversation_templates = self.load_conversation_templates(self.conversation_path)
        self.mask_to_polygons = MaskPolygonConverter()

        # 添加随机控制器
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self._idx = 0
        self._idx_list = self.random_state.permutation(len(self.data)).tolist()

        self.with_instruct = True
        logger.warning("是否使用文本指导信息: {}".format(self.with_instruct))
        self.special_token = (
            DEFAULT_CLASS_TOKEN + DEFAULT_POS_TOKEN + "".join([DEFAULT_MASK_TOKEN.format(i=i) for i in range(self.num_new_tokens)])
        )
        self.set_system_message()

    def __len__(self):
        return len(self.data)

    def data_processing(self, index):
        # 获取json文件
        if len(self.data[index]) == 4:
            img_path, mask_path, gt_label, instruct = self.data[index]
        elif len(self.data[index]) == 3:
            img_path, mask_path, gt_label = self.data[index]
            instruct = None
        else:
            img_path, mask_path = self.data[index]
            gt_label = 0 if mask_path == "positive" else 1
            instruct = None

        # 处理图片
        try:
            image = self.load_image(img_path)
        except Exception as e:
            logger.error(f"Error in loading image: {e}. image: {img_path}, mask: {mask_path}, label: {gt_label}")
            raise e

        if mask_path is None:
            mask = None
            ret = self.common_transform(image=np.array(image), mask=None)
            pos_ret = self.post_transform(image=ret["image"], mask=None)
        else:
            if mask_path == "positive":
                mask = Image.new("L", image.size, 0)
            else:
                mask = self.load_image(mask_path, "L")
            ret = self.common_transform(image=np.array(image), mask=np.array(mask))
            pos_ret = self.post_transform(image=ret["image"], mask=ret["mask"])

        np_img, np_mask = pos_ret["image"], pos_ret.get("mask", None)
        return (img_path, mask_path, np_img, np_mask, gt_label, instruct)

    def print_info(self):
        logger.info("===" * 15)
        logger.info(f"Dataset Info: {self.dataset_path}")
        logger.info(f"Number of samples: {len(self.data)}")
        logger.info(f"Number of templates: {len(self.conversation_templates)}")
        logger.info(f"Number of new tokens: {self.num_new_tokens}")

    def get_expert_info(self, message, flag="<expert>"):
        """获取专家信息并返回处理后的消息"""
        expert_info = "".join([DEFAULT_EXPERT_TOKEN.format(i=i) for i in range(self.num_new_tokens)])
        if flag in message:
            processed_message = message.replace(flag, expert_info)
        else:
            processed_message = expert_info + message

        return processed_message

    def set_system_message(self):
        """设置系统消息
        设置不设置 system message 对结果的影响
        """
        # 替换system message中的标识符
        system_msg = self.encode_func.template_meta.default_system
        system_msg = system_msg.replace("<SPECIAL_TOKEN>", self.special_token)
        self.encode_func.template_meta.default_system = system_msg
        logger.info(f"重设system msg的内容: {system_msg}")

    def get_image_pos_info(self, mask, label, is_ploygons=True, is_bbox=True, *args, **kwargs):
        """获取图像位置的信息"""
        if label == 0 or mask is None:
            positions = []
        elif is_ploygons:
            positions = self.mask_to_polygons.convert_to_polygons(mask)
        elif is_bbox:
            positions = self.mask_to_polygons.convert_to_bboxes(mask, self.split_box_num, *args, **kwargs)
        else:
            raise ValueError("is_ploygons and is_bbox are both False")

        if len(positions) == 0 and label != 0 and mask is not None:
            raise ValueError("No valid polygons found in the fake mask.")

        # ploygons前加上特殊的标识符
        pos_str = ",".join(str(pos) for pos in positions)
        return f"[{pos_str}]", positions

    def get_conversation(self, mask, label, instruct, result_separator="<res>"):
        # 选择模板
        conversation_template = self.random_state.choice(self.conversation_templates)
        assistant_message = conversation_template["assistant"]
        assert result_separator in assistant_message, f"Assistant message not contain result_separator: {result_separator}"

        # 执行对话
        gt_text = OrderedDict()

        # 二分类效果标注
        label_info = "real" if label == 0 else "fake"
        pos_info, positions = self.get_image_pos_info(mask, label, is_ploygons=False, is_bbox=True)

        gt_text["result"] = label_info
        gt_text["mask"] = DEFAULT_BOX_START_TOKEN + pos_info + DEFAULT_BOX_END_TOKEN
        gt_info = json.dumps(gt_text, ensure_ascii=False)[:-1]
        if self.token_pos == TokenPosEnum.FRONT:
            gt_info = self.special_token + gt_info
        elif self.token_pos == TokenPosEnum.BACK:
            gt_info = gt_info + self.special_token

        # 文本提示内容标注
        if self.with_instruct:
            if not instruct:
                instruct = ""
            gt_info = gt_info + f', "reason": "{instruct}"'

        gt_info = gt_info + "}"
        assistant_message = conversation_template["assistant"].replace(result_separator, gt_info)

        # 加入特殊标识
        user_message = conversation_template["user"]
        if self.add_expert_feat:
            user_message = self.get_expert_info(user_message)
        return user_message, assistant_message, positions

    def __getitem__(self, idx):
        # 选择模板
        try:
            img_path, mask_path, gt_img, gt_mask, gt_label, instruct = self.data_processing(idx)
            user_msg, assist_msg, gt_boxes = self.get_conversation(gt_mask, gt_label, instruct)
            # np 转为 PIL
            # 生成数据
            data = {
                "images": [{"bytes": Image.fromarray(gt_img)}],
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assist_msg},
                ],
                "image_path": img_path,
                "mask_path": mask_path,
                "gt_img": gt_img,
                "gt_label": gt_label,
                "gt_boxes": gt_boxes,
            }
            if self.pred_mask:
                # 将mask转为0和1, 使用阈值判断
                data["gt_mask"] = gt_mask.astype(float) / 255.0 if gt_mask is not None else None
            if self.encode_func is None:
                return data
            return self.encode_func.encode(data, return_template_inputs=True)
        except Exception as e:
            logger.error(f"Error in get_conversation: {e}. image: {img_path}, mask: {mask_path}, label: {gt_label}")
            raise e

    def check_data(self):
        # 随机取一个item, 检测是否能够正常运行
        try:
            self.__getitem__(self._idx_list[0])
            return True
        except Exception as e:
            return False
