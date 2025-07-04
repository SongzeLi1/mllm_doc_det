import json
from pathlib import Path


def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def read_content(file_path):
    with open(file_path, "r") as file:
        return file.read()


def opt_magic():
    magic_json = read_json("/pubdata/yuyangxin/swift-demo/resource/datasets/without_instruct/MagicBrush_match.json")
    ret = []
    for i in magic_json:
        if i[2] == 0:
            continue
        ret.append(i)

    save_path = "./magic.json"
    with open(save_path, "w") as file:
        json.dump(ret, file, indent=4)


def main():
    target_dir = Path("/home/yuyangxin/data/dataset/MagicBrush/images")
    mask_dir = Path("/home/yuyangxin/data/dataset/MagicBrush/fake_mask")
    # 遍历目录下所有文件夹
    ret = []
    for folder in target_dir.iterdir():
        if folder.is_dir():
            # 遍历文件夹内所有文件
            real_file = None
            fake_file = None
            for file in folder.iterdir():
                if file.is_file():
                    if file.name.endswith("input.png"):
                        real_file = file
                    elif file.name.endswith("output1.png"):
                        fake_file = file

            # 获取文件名, 没有_mask.png后缀
            file_name = fake_file.name.replace("-output1", "_mask")
            # 查找目标文件夹下是否存在对应的mask文件
            mask_file = mask_dir / file_name
            if not mask_file.exists():
                raise ValueError(f"Mask file {mask_file} not found")

            ret.extend(
                [
                    [real_file.absolute().as_posix(), "positive", 0],
                    [fake_file.absolute().as_posix(), mask_file.absolute().as_posix(), 1],
                ]
            )
    # 将结果写入json文件
    with open("./magic.json", "w") as file:
        json.dump(ret, file, indent=4)


if __name__ == "__main__":
    opt_magic()
