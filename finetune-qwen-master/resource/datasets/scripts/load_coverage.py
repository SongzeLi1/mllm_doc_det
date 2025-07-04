import json
from pathlib import Path

if __name__ == "__main__":
    # 1. 读取数据集路径
    # 2. 遍历数据集中的所有图片
    # 3. 将图片路径和标签保存到json文件中
    target_str = "IMD2020"
    target_path = Path(f"/home/yuyangxin/data/dataset/{target_str}")

    au_path = target_path / "Au"
    ret = []
    for path in au_path.glob("*.jpg"):
        ret.append(
            [
                path.absolute().as_posix(),
                "positive",
                0,
            ]
        )

    tp_path_dir = target_path / "Tp"
    gt_path_dir = target_path / "Gt"
    for path in tp_path_dir.glob("*.png"):
        gt_path = gt_path_dir / path.name
        if not gt_path.exists():
            raise ValueError(f"Ground truth file {gt_path} does not exist")
        ret.append(
            [
                path.absolute().as_posix(),
                gt_path.absolute().as_posix(),
                1,
            ]
        )

    # 将ret保存
    save_path = f"/home/yuyangxin/data/finetune-qwen/resource/datasets/without_instruct/{target_str}.json"
    with open(save_path, "w") as f:
        json.dump(ret, f, indent=4)
