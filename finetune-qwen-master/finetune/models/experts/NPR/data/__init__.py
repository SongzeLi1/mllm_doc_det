import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder

"""
def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)
"""

import os


def get_dataset(opt):
    classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
    if "0_real" not in classes or "1_fake" not in classes:
        dset_lst = []
        for cls in classes:
            root = opt.dataroot + "/" + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    return dataset_folder(opt, opt.dataroot)


def get_bal_sampler(dataset):
    # 生成加权随机采样器，用于类平衡
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,  # 允许用户自定义从数据集中采样的策略。
        num_workers=int(opt.num_threads),
    )

    # 展示数据集信息
    print(f"dataset length: {len(data_loader.dataset)}")
    # 可视化一个batch的图片
    visualize_batch(data_loader)
    return data_loader


import matplotlib.pyplot as plt


def visualize_batch(data_loader):
    # 获取一个batch的数据
    batch = next(iter(data_loader))

    # 假设batch是一个元组 (images, labels)
    images, labels = batch

    # 如果图像是张量，转换为numpy数组
    if isinstance(images, torch.Tensor):
        images = images.numpy()

    # 假设图像是 (batch_size, channels, height, width)
    # 将图像转换为 (height, width, channels) 格式
    images = np.transpose(images, (0, 2, 3, 1))

    # 创建一个图形窗口
    fig, axes = plt.subplots(1, min(len(images), 4), figsize=(12, 3))

    if len(images) == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one image

    for img, ax in zip(images, axes):
        ax.imshow(img.astype(np.uint8))  # Assuming the image is in 0-255 range
        ax.axis("off")

    # 保存图片
    plt.savefig("image.png")
