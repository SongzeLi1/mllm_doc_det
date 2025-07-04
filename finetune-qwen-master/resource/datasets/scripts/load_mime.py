import json
from pathlib import Path

path_info = Path("/pubdata/yuyangxin/swift-demo/resource/datasets/mime.json")
save_path = Path("/pubdata/yuyangxin/swift-demo/resource/datasets/without_instruct/MIML_real.json")

# 读取json文件
with open(path_info, "r", encoding="utf-8") as f:
    datas = json.load(f)

# 读取每一个数据, 如果label为0, 则将其删除
ret = []
for data in datas:
    fake_image, gt_mask, label = data[0], data[1], data[2]
    if label == 1:
        continue
    ret.append([fake_image, gt_mask, label])

# 将数据写入json文件
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(ret, f, ensure_ascii=False, indent=4)
