import json
import os
import random
import re
import shutil


def opt_info(texts: list):
    # 切分数据的内容, 提取 'image:' 和 ',mask:' 之间的内容
    results = []

    for text in texts:
        # 使用正则表达式匹配 'image:' 和 ',mask:' 之间的内容
        matches = re.findall(r"image:(.*?), mask:", text, re.DOTALL)

        # 将匹配到的内容添加到结果列表中
        for match in matches:
            results.append(match.strip())
    return results


def get_json_data(json_path: str, info: list):
    # 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取info的内容
    info = opt_info(info)
    # 提取 'image' 和 'mask' 的路径
    new_info = []
    for item in data:
        if item[0] in info:
            print(f"image: {item[0]}, mask: {item[1]}")
            continue
        new_info.append(item)

    # 将数据随机打乱
    random.shuffle(new_info)
    # 将打乱后的数据写入新的 JSON 文件
    new_json_path = os.path.join(os.path.dirname(json_path), "new_data.json")
    with open(new_json_path, "w", encoding="utf-8") as f:
        json.dump(new_info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)

    # 定义输入和输出路径
    input_json_path = "/home/yuyangxin/data/finetune-qwen/resource/datasets/without_instruct/MIML_Part1_fake.json"  # 输入 JSON 文件路径

    info = """2025-04-25 13:47:11 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/6692.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/6692.png, label: 1
处理数据项 3119 时出错: No valid polygons found in the fake mask.
2025-04-25 13:47:35 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/239.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/239.png, label: 1
处理数据项 6396 时出错: No valid polygons found in the fake mask.
2025-04-25 13:47:37 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/40475.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/40475.png, label: 1
处理数据项 6605 时出错: No valid polygons found in the fake mask.
2025-04-25 13:47:38 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/43114.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/43114.png, label: 1
处理数据项 6717 时出错: No valid polygons found in the fake mask.
2025-04-25 13:47:39 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/30165.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/30165.png, label: 1
处理数据项 6907 时出错: No valid polygons found in the fake mask.
2025-04-25 13:47:58 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/32326.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/32326.png, label: 1
处理数据项 9270 时出错: No valid polygons found in the fake mask.
2025-04-25 13:48:06 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/12133.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/12133.png, label: 1
处理数据项 10200 时出错: No valid polygons found in the fake mask.
2025-04-25 13:48:37 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/21466.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/21466.png, label: 1
处理数据项 13658 时出错: No valid polygons found in the fake mask.
2025-04-25 13:48:50 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/32002.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/32002.png, label: 1
处理数据项 15186 时出错: No valid polygons found in the fake mask.
2025-04-25 13:48:55 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/33861.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/33861.png, label: 1
处理数据项 15698 时出错: No valid polygons found in the fake mask.
2025-04-25 13:49:00 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/25896.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/25896.png, label: 1
处理数据项 16167 时出错: No valid polygons found in the fake mask.
2025-04-25 13:49:04 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/26102.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/26102.png, label: 1
处理数据项 16621 时出错: No valid polygons found in the fake mask.
2025-04-25 13:49:11 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/5154.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/5154.png, label: 1
处理数据项 17266 时出错: No valid polygons found in the fake mask.
2025-04-25 13:49:12 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/6341.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/6341.png, label: 1
处理数据项 17422 时出错: No valid polygons found in the fake mask.
2025-04-25 13:50:02 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/14149.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/14149.png, label: 1
处理数据项 22062 时出错: No valid polygons found in the fake mask.
2025-04-25 13:50:22 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/25083.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/25083.png, label: 1
处理数据项 23875 时出错: No valid polygons found in the fake mask.
2025-04-25 13:50:44 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/21105.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/21105.png, label: 1
处理数据项 25666 时出错: No valid polygons found in the fake mask.
2025-04-25 13:51:13 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/2401.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/2401.png, label: 1
处理数据项 26345 时出错: No valid polygons found in the fake mask.
2025-04-25 13:51:14 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/7173.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/7173.png, label: 1
处理数据项 26367 时出错: No valid polygons found in the fake mask.
2025-04-25 13:52:00 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/14311.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/14311.png, label: 1
处理数据项 27472 时出错: No valid polygons found in the fake mask.
2025-04-25 13:52:28 - ERROR - Error in get_conversation: No valid polygons found in the fake mask.. image: /home/yuyangxin/data/dataset/MIML/MIML_Part1/imgs/36616.jpg, mask: /home/yuyangxin/data/dataset/MIML/MIML_Part1/masks/36616.png, label: 1
处理数据项 28348 时出错: No valid polygons found in the fake mask.
    """
    # 将info字符串按行分割成列表
    info = info.strip().split("\n")
    # 调用函数处理数据
    get_json_data(input_json_path, info)
