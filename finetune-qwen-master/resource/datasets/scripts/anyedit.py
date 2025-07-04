# 读取文件夹下所有得文件夹
import json
from pathlib import Path


def main():
    # 读取json的内容
    target_path = Path(r"/data0/yuyangxin/finetune-qwen/resource/datasets/without_instruct/anybench.json")
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ret = []
    for i in data:
        if i[2] == 1:
            ret.append(i)
    # 将ret的内容写入到文件中
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(ret, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
