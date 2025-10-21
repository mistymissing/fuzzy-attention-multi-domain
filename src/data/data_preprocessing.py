# -*- coding: utf-8 -*-
"""
数据预处理脚本 (Data Preprocessing)
-----------------------------------
功能:
1. 检查四个数据集 (CIFAR-10 / IMDB / NSL-KDD / Chest X-ray)
2. 支持简单的数据预处理 (后续扩展)
3. 命令行参数 --check: 只检查数据是否存在和大小

数据目录默认结构:
data/
    cifar-10-batches-py/
    imdb/
    nsl-kdd/
    chest_xray_pneumonia/

注意:
Windows 下路径请使用正斜杠 "/" 或者双反斜杠 "\\"。
"""

import os
import argparse

def check_datasets(data_root: str):
    """检查四个数据集是否存在，并打印基本信息"""
    datasets = {
        "CIFAR-10": os.path.join(data_root, "cifar-10-batches-py"),
        "IMDB": os.path.join(data_root, "imdb"),
        "NSL-KDD": os.path.join(data_root, "nsl-kdd"),
        "Chest X-ray": os.path.join(data_root, "chest_xray_pneumonia"),
    }

    print(f"📂 数据根目录: {data_root}\n")
    for name, path in datasets.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                total_files = sum(len(files) for _, _, files in os.walk(path))
                print(f"✅ {name:<10} 已找到, 目录: {path}, 文件数: {total_files}")
            else:
                size = os.path.getsize(path) / 1024 / 1024
                print(f"✅ {name:<10} 已找到, 文件大小: {size:.2f} MB")
        else:
            print(f"❌ {name:<10} 未找到, 期望路径: {path}")

def main():
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
    parser.add_argument("--check", action="store_true", help="仅检查数据集是否存在")
    args = parser.parse_args()

    if args.check:
        check_datasets(args.data_root)

if __name__ == "__main__":
    main()
