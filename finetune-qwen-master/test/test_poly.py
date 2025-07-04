# import sys
# from pathlib import Path

# # 添加父目录到 Python 路径
# sys.path.append("..")
# import pytest
# import numpy as np
# from finetune.utils.polygon_convert import MaskPolygonConverter

# # python


# def box_set(bboxes):
#     return {tuple(b) for b in bboxes}


# @pytest.fixture
# def converter():
#     return MaskPolygonConverter()


# def test_empty_mask(converter):
#     mask = np.zeros((5, 5), dtype=np.uint8)
#     assert converter.convert_to_bboxes(mask) == []


# def test_full_mask(converter):
#     h, w = 4, 6
#     mask = np.full((h, w), 255, dtype=np.uint8)
#     res = converter.convert_to_bboxes(mask)
#     assert res == [[0, 0, w, h]]


# def test_single_pixel(converter):
#     mask = np.zeros((5, 5), dtype=np.uint8)
#     mask[2, 3] = 255
#     # max_boxes default = 1
#     res = converter.convert_to_bboxes(mask)
#     assert res == [[3, 2, 4, 3]]


# def test_two_pixels_two_boxes(converter):
#     mask = np.zeros((5, 5), dtype=np.uint8)
#     pts = [(1, 1), (3, 4)]
#     for x, y in pts:
#         mask[x, y] = 255
#     res = converter.convert_to_bboxes(mask, max_boxes=2)
#     expected = {(1, 1, 2, 2), (3, 4, 4, 5)}
#     assert box_set(res) == expected


# def test_two_pixels_one_box(converter):
#     mask = np.zeros((5, 5), dtype=np.uint8)
#     mask[0, 2] = 255
#     mask[4, 1] = 255
#     # max_boxes < 2
#     res = converter.convert_to_bboxes(mask, max_boxes=1)
#     # boundingRect covers from x=1 to x=3, y=0 to y=5
#     assert res == [[1, 0, 3, 5]]


# def test_binarize_false_same_as_true(converter):
#     mask = np.zeros((5, 5), dtype=np.uint8)
#     mask[0, 0] = 128
#     # threshold at 127 → binarize True sees that pixel, binarize False keeps 128 (>0)
#     res_true = converter.convert_to_bboxes(mask, max_boxes=1, binarize=True)
#     res_false = converter.convert_to_bboxes(mask, max_boxes=1, binarize=False)
#     assert res_true == res_false == [[0, 0, 1, 1]]
