import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def process_info(info):
    if info["label"] == 0:
        img_save_path = natural_img_dir / f"{info['img_id']}.png"
        gt_path = "positive"
        label = 0
    elif info["label"] == 2:
        img_save_path = tp_save_dir / f"{info['img_id']}.png"
        gt_path = mask_save_dir / f"{info['img_id']}.png"
        gt_path = gt_path.absolute().as_posix()
        label = 1
    else:
        return None

    img = info["image"]
    if img.mode == "CMYK":
        img = img.convert("RGB")
    img.save(img_save_path)
    if info["mask"] and gt_path != "positive":
        mask = info["mask"]
        if mask.mode == "CMYK":
            mask = mask.convert("RGB")
        if mask.size != img.size:
            print(f"Resizing mask from {mask.size} to {img.size}")
            mask = mask.resize(img.size)
        mask.save(gt_path)
    return [
        img_save_path.absolute().as_posix(),
        gt_path,
        label,
    ]


# 使用 streaming=True 流式读取
dataset = load_dataset("parquet", data_dir="/data2/public_data/SID_Set", streaming=True)

# train: 210k rows
# test: 30k rows

# 切分test, 要求仅保留 img_id以"tampered"开头的行, 并将切分的数据集保存到文件夹下
count = 0
save_dir = Path("/data1/yuyangxin/datasets")
save_dir.mkdir(parents=True, exist_ok=True)

natural_img_dir = save_dir / "SIDA_validation" / "Au"
natural_img_dir.mkdir(parents=True, exist_ok=True)
tp_save_dir = save_dir / "SIDA_validation" / "Tp"
tp_save_dir.mkdir(parents=True, exist_ok=True)
mask_save_dir = save_dir / "SIDA_validation" / "Gt"
mask_save_dir.mkdir(parents=True, exist_ok=True)

ret = []
tqdm_iter = tqdm(dataset["validation"], desc="Processing validation set")
with ThreadPoolExecutor(max_workers=32) as executor:  # 可根据CPU调整线程数
    futures = [executor.submit(process_info, info) for info in tqdm_iter]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Saving images"):
        result = future.result()
        if result is not None:
            ret.append(result)


# 写入json文件
save_json_path = save_dir / "sida_validation_with_mask.json"
with open(save_json_path, "w") as f:
    json.dump(ret, f, indent=4)

print(f"Saved {len(ret)} images to {save_json_path}")
print("Done")
