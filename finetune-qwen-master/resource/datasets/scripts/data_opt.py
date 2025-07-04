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


def read_content(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_sub_dataset(total_json_path, sub_json_path):
    total_data = read_json(total_json_path)
    sub_data = read_json(sub_json_path)
    # 获取total_data中sub_data中不存在的数据
    ret = set([i[0] for i in total_data]) - set([i[0] for i in sub_data])
    ret = list(ret)
    # 保存到文件
    save_res = []
    for i in total_data:
        if i[0] in ret:
            save_res.append(i)
    print(f"Total {len(save_res)} images")
    save_path = (
        Path("/home/yuyangxin/data/finetune_LLM/resources/datasets/sub_dataset")
        / f"{Path(total_json_path).stem}_sub.json"
    )
    print(f"Save to {save_path}")
    with open(save_path, "w") as file:
        json.dump(save_res, file, indent=4)
    return save_path


def write_json(target_file, instructions):

    target_data = read_json(target_file)
    save_path = Path("/home/yuyangxin/data/finetune_LLM/resources/datasets")
    instructions = Path(instructions)
    ret = []
    for i in target_data:
        image_path, mask_path = Path(i[0]), Path(i[1])
        instruct = instructions / f"{image_path.stem}.txt"
        if not instruct.exists():
            print(f"{instruct} does not exist, skip")
            continue

        assert mask_path.exists(), f"{mask_path} does not exist"
        # 打开mask文件,检测像素值是否全为0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        fake_instruct = read_content(instruct)
        ret.append(
            {
                "image_path": i[0],
                "mask_path": i[1],
                "label": 0 if mask.sum() == 0 else 1,
                "instruct": fake_instruct,
            }
        )
    print(f"Total {len(ret)} images")
    with open(save_path / f"{target_file.stem}_fake_with_instruct.json", "w") as file:
        json.dump(ret, file, indent=4)


def process_single_item(item):
    """处理单个数据项的函数"""
    img_path, mask_path = Path(item[0]), Path(item[1])
    # 判断mask是否是全白的
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask.sum() == 0:
        label = 1
    else:
        label = 0
    return (str(img_path), str(mask_path), label)


def operation_data(data_path, max_workers=16):
    # 操作数据集
    data = read_json(data_path)

    # 使用线程池并行处理数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并获取future对象
        futures = [executor.submit(process_single_item, item) for item in data]

        # 使用tqdm显示进度
        new_data = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理图像"):
            new_data.append(future.result())

    print(f"共处理了 {len(new_data)} 个图像")

    # 写回JSON文件
    with open(data_path, "w") as file:
        json.dump(new_data, file, indent=4)


def operation_(data_path):
    data = read_json(data_path)
    save_target = Path("/home/yuyangxin/data/finetune_LLM/resources/datasets/without_instruct")
    new_data = []
    for key, values in data.items():
        real_info = (values["real_img"], "Negative", 0)
        new_data.append(real_info)
        fake_info = (values["fake_img"], values["fake_mask"], 1)
        new_data.append(fake_info)
    with open(save_target / data_path.name, "w") as file:
        json.dump(new_data, file, indent=4)


def process_data_item(item):
    """处理单个数据项的函数"""
    if len(item) == 3:
        img_path, mask_path, label = item[0], item[1], item[2]
        reason = "This is a real image, with textures, lighting, etc. that match natural images"
    else:
        img_path, mask_path, label, reason = item[0], item[1], item[2], item[3]

    # 替换路径
    a = img_path.split("/data0/yuyangxin/dataset")
    if len(a) == 2:
        img_path = "/pubdata/yuyangxin/dataset" + a[1]
    a = mask_path.split("/data0/yuyangxin/dataset")
    if len(a) == 2:
        mask_path = "/pubdata/yuyangxin/dataset" + a[1]

    # 读取mask_path, 判断是否是全黑的
    if mask_path == "Negative":
        if label == 1:
            print(f"{img_path} is real image, but label is 1")
            label = 0
        mask_path = "positive"
    else:
        try:
            mask = np.array(Image.open(mask_path).convert("L"))
            # 如果mask是全黑的,则label置为0
            if mask.sum() == 0:
                if label == 1:
                    print(f"{mask_path} mask全黑, but label is 1")
                    label = 0
        except Exception as e:
            raise ValueError(f"处理 {mask_path} 时发生错误: {e}")

    return (img_path, mask_path, label, reason)


def data_clean(data_path, max_workers=32):
    """多线程并发清理数据"""
    data = read_json(data_path)

    # 使用线程池并行处理数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并获取future对象
        futures = [executor.submit(process_data_item, item) for item in data]

        # 使用tqdm显示进度
        new_data = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="清理数据"):
            res = future.result()
            if res[-1] is not None:
                new_data.append(res)
            else:
                new_data.append(res[:-1])

    print(f"共处理了 {len(new_data)} 个数据项")

    # 写回JSON文件
    with open(data_path, "w") as file:
        json.dump(new_data, file, indent=4)


def data_check2(data_path):
    data = read_json(data_path)
    new_data = []
    for item in data:
        img_path, mask_path, label = item
        if mask_path == "positive":
            if "output" in img_path:
                print(f"[Error]: {img_path} is positive, but have output")
                mask_path = Path(img_path.replace("output", "mask"))
                if not mask_path.exists():
                    print(f"[Warning]: {mask_path} does not exist")
                    label = 1
                    continue
                else:
                    mask_path = mask_path.absolute().as_posix()
                    label = 1
        else:
            if "output" not in img_path:
                raise ValueError(f"[Error]: {img_path} is negative, but without output")
        new_data.append((img_path, mask_path, label))
    # 写回JSON文件
    with open(data_path, "w") as file:
        json.dump(new_data, file, indent=4)


if __name__ == "__main__":
    # target_file = Path("/home/yuyangxin/data/dataset/MIML/match_MIML_Part2.json")
    # instructions = Path("/home/yuyangxin/data/finetune_LLM/resources/ForgeryAnalysis-PT/MIML/MIML_Part2")
    # write_json(target_file, instructions)
    # operation_data("/home/yuyangxin/data/dataset/MIML/match_MIML_Part1.json")
    # operation_data("/home/yuyangxin/data/dataset/MIML/match_MIML_Part2.json")
    # operation_(Path("/home/yuyangxin/data/dataset/MagicBrush/record.json"))
    # get_sub_dataset(
    #     "/home/yuyangxin/data/finetune_LLM/resources/datasets/without_instruct/match_MIML_Part1.json",
    #     "/home/yuyangxin/data/finetune_LLM/resources/datasets/with_instruct/match_MIML_Part1_fake_with_instruct.json",
    # )
    # data_clean("/pubdata/yuyangxin/swift-demo/resource/datasets/sub_dataset/CASIAv2.json_sub.json")
    # data_clean("/pubdata/yuyangxin/swift-demo/resource/datasets/without_instruct/match_MIML_Part2.json")
    data_check2("/pubdata/yuyangxin/swift-demo/resource/datasets/without_instruct/MagicBrush.json")
