from pathlib import Path

save_path = Path("/pubdata/yuyangxin/swift-demo/resource/datasets/without_instruct/CASIAv1_real.json")
target_path = Path("/home/yuyangxin/data/dataset/CASIAv1/Au")
# 读取目标文件下的所有以 .jpg 结尾的文件
files = list(target_path.glob("**/*.jpg"))
ret = []
for file in files:
    ret.append([file.as_posix(), "positive", 0])

print(f"total {len(ret)} files")

# 保存到目标路径
import json

with open(save_path, "w") as f:
    json.dump(ret, f, indent=4)
