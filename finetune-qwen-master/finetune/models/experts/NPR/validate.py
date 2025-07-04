from pathlib import Path
import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score
from options.test_options import TestOptions
from data import create_dataloader
import json
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# 定义您的选项类，其中包含所需的参数
class Options:
    def __init__(self, isTrain=True, cropSize=224, no_crop=False, no_flip=False, no_resize=False, loadSize=256):
        self.isTrain = isTrain  # 是否是训练模式
        self.cropSize = cropSize  # 裁剪大小
        self.no_crop = no_crop  # 是否不进行裁剪
        self.no_flip = no_flip  # 是否不进行翻转
        self.no_resize = no_resize  # 是否不进行缩放
        self.loadSize = loadSize  # 缩放大小


# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, json_file, opt):
        # 读取json文件
        with open(json_file, "r") as f:
            self.samples = json.load(f)

        self.opt = opt
        # 定义图像转换
        if self.opt.isTrain:
            crop_func = transforms.RandomCrop(self.opt.cropSize)
        elif self.opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(self.opt.cropSize)

        if self.opt.isTrain and not self.opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)

        if not self.opt.isTrain and self.opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Resize((self.opt.loadSize, self.opt.loadSize))

        self.transform = transforms.Compose(
            [
                rz_func,
                # 您可以在此添加其他数据增强操作，例如：
                # transforms.Lambda(lambda img: custom_resize(img, self.opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取图像路径和标签字符串
        image_path, label_str = self.samples[idx]
        # 打开图像
        image = Image.open(image_path).convert("RGB")
        # 应用图像转换
        image = self.transform(image)
        # 根据标签字符串设置标签
        if label_str == "Negative":
            label = 0
        else:
            label = 1
        return image, label


def validate(model, json_file, opt):
    # data_loader = create_dataloader(opt)
    # 创建数据集
    dataset = CustomImageDataset(json_file, opt)
    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    # f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    f1 = f1_score(y_true, y_pred > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    # ap = average_precision_score(y_true, y_pred)
    return acc, f1, y_true, y_pred


if __name__ == "__main__":
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    data_dir = Path("/home/yuyangxin/data/imdl-demo/datasets/dragdiff")
    json_list = data_dir.glob("*.json")
    for i in json_list:
        print("测试集:", i)
        acc, f1, y_true, y_pred = validate(model, i, opt)
        print("Accuracy:", acc, "F1:", f1)
