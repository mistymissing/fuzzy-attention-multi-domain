"""
dataset_loader.py
四域统一数据加载器：
- Vision: CIFAR-10（或本地自定义图片文件夹）
- NLP: 任意CSV文本分类数据集（两列：text,label）
- Security: NSL-KDD风格CSV（数值特征列 + label 列）
- Medical: ChestX-ray14风格（images 目录 + labels.csv: image,label1,...）

返回值统一为 AdapterInput：
AdapterInput(raw_data=..., labels=..., metadata={...})
"""

import os
import csv
import math
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# 你项目里的基类（和适配器共用）
from base_adapter import AdapterInput, AdapterMode

# --- 可选依赖 ---
# Vision
try:
    from torchvision import datasets as tvds, transforms
except Exception:
    tvds, transforms = None, None

# Medical
from PIL import Image
import numpy as np
import pandas as pd


# ---------------------------
# 通用配置与小工具
# ---------------------------

@dataclass
class LoaderConfig:
    # 公共
    batch_size: int = 16
    num_workers: int = 0   # Win/IDE 环境建议 0，Linux 可设 4/8
    shuffle_train: bool = True
    pin_memory: bool = torch.cuda.is_available()

    # Vision
    vision_root: str = "./datasets/cifar10"
    vision_use_cifar10: bool = True  # 否则用 ImageFolder
    vision_imagefolder_root: str = "./datasets/vision_images"  # 当不使用 CIFAR10 时

    # NLP
    nlp_csv_path: str = "./datasets/imdb/train.csv"  # 两列: text,label
    nlp_val_csv_path: Optional[str] = None
    nlp_lower: bool = True

    # Security
    sec_csv_path: str = "./datasets/nsl_kdd/nsl_kdd.csv"  # 包含 label 列
    sec_label_col: str = "label"
    sec_ignore_cols: Tuple[str, ...] = ("protocol_type", "service", "flag")  # 若存在这些离散列，可先忽略
    sec_limit: Optional[int] = None  # 子集调试

    # Medical
    med_image_root: str = "./datasets/chestxray14/images"
    med_label_csv: str = "./datasets/chestxray14/labels.csv"  # 两列或多列: image, label(s)...
    med_binary: bool = True  # True: 单病种/二分类；False: 多标签
    med_label_cols: Optional[List[str]] = None  # 多标签列名（med_binary=False 时使用）
    med_resize: int = 224


# ---------------------------
# VISION
# ---------------------------

class VisionCIFAR10Dataset(Dataset):
    """
    默认使用 CIFAR-10（自动下载到 vision_root）。
    如果你想用自己的图片文件夹（class 子文件夹结构），把 config.vision_use_cifar10 = False，
    并把图片放到 config.vision_imagefolder_root 下。
    """

    def __init__(self, config: LoaderConfig, train: bool = True):
        assert transforms is not None, "需要安装 torchvision 才能加载图像数据。"

        self.config = config

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        if config.vision_use_cifar10:
            os.makedirs(config.vision_root, exist_ok=True)
            self.ds = tvds.CIFAR10(
                root=config.vision_root,
                train=train,
                download=True,
                transform=self.tf
            )
            self.num_classes = 10
        else:
            # 使用 ImageFolder（支持任意自定义图片数据）
            from torchvision.datasets import ImageFolder
            root = os.path.join(config.vision_imagefolder_root, "train" if train else "val")
            self.ds = ImageFolder(root=root, transform=self.tf)
            self.num_classes = len(self.ds.classes)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]  # img: Tensor[C,H,W] float
        return AdapterInput(
            raw_data=img,                          # 直接给 VisionAdapter
            labels=torch.tensor(label, dtype=torch.long),
            metadata={"domain": "vision"}
        )


# ---------------------------
# NLP
# ---------------------------

class NLPTextCSVDataset(Dataset):
    """
    适配任意 CSV 文本分类数据（两列：text,label）
    文件示例:
        text,label
        "I love this movie",2
        "awful acting",0
    """
    def __init__(self, csv_path: str, lower: bool = True):
        self.samples: List[Tuple[str, int]] = []
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"NLP CSV 不存在: {csv_path}")

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            assert "text" in reader.fieldnames and "label" in reader.fieldnames, \
                "CSV 需要包含列名 text 与 label"
            for row in reader:
                text = row["text"] or ""
                if lower:
                    text = text.lower()
                label = int(row["label"])
                self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        # NLPAdapter 会自己 tokenize，这里直接给字符串
        return AdapterInput(
            raw_data=text,
            labels=torch.tensor(label, dtype=torch.long),
            metadata={"domain": "nlp"}
        )


# ---------------------------
# SECURITY
# ---------------------------

class SecurityNSLKDDDataset(Dataset):
    """
    NSL-KDD 风格的 CSV。默认把非数值列（protocol_type/service/flag）忽略。
    你也可以把它们先做 one-hot 后再合并，这里为了最小可用先忽略。
    需要有一个 'label' 列（整数标签或字符串标签）。
    """
    def __init__(self, csv_path: str, label_col: str = "label",
                 ignore_cols: Tuple[str, ...] = ("protocol_type","service","flag"),
                 limit: Optional[int] = None):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Security CSV 不存在: {csv_path}")

        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.head(limit)

        assert label_col in df.columns, f"CSV必须包含列: {label_col}"

        # 选择数值特征
        numeric_cols = [c for c in df.columns if c != label_col and c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
        x = df[numeric_cols].values.astype(np.float32)
        y = df[label_col].values

        # 若 label 是字符串，则简单 map 到整数
        if not np.issubdtype(y.dtype, np.integer):
            classes = {v:i for i,v in enumerate(sorted(set(y.tolist())))}
            y = np.array([classes[v] for v in y], dtype=np.int64)

        self.x = torch.from_numpy(x)             # [N, D]
        self.y = torch.from_numpy(y)             # [N]
        self.numeric_cols = numeric_cols

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        feats = self.x[idx]       # Tensor[D], float32
        label = self.y[idx]       # Tensor[]
        return AdapterInput(
            raw_data=feats,       # SecurityAdapter 支持 Tensor/ndarray/df
            labels=label,
            metadata={"domain": "security"}
        )


# ---------------------------
# MEDICAL
# ---------------------------

class MedicalXrayCSVDataset(Dataset):
    """
    ChestX-ray14 风格：
      - 图像目录: med_image_root
      - 标注CSV: med_label_csv
        * 二分类: 两列 [image, label]，label ∈ {0,1}
        * 多标签: 多列 [image, pathology1, pathology2, ...] (0/1)
    """
    def __init__(self,
                 image_root: str,
                 label_csv: str,
                 resize: int = 224,
                 binary: bool = True,
                 label_cols: Optional[List[str]] = None):
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Medical 图像目录不存在: {image_root}")
        if not os.path.isfile(label_csv):
            raise FileNotFoundError(f"Medical labels CSV 不存在: {label_csv}")

        self.image_root = image_root
        self.df = pd.read_csv(label_csv)
        self.binary = binary
        self.label_cols = label_cols

        if self.binary:
            assert "image" in self.df.columns and "label" in self.df.columns, \
                "二分类 CSV 必须包含列: image,label"
        else:
            assert "image" in self.df.columns and label_cols is not None and len(label_cols) > 0, \
                "多标签 CSV 必须包含列: image 以及显式的 label_cols"

        # 视觉预处理（只做尺寸与归一化，颜色通道由适配器里处理/适配）
        self.resize = resize

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        """
        转成 [C,H,W] float32, 0~1 范围。灰度或RGB都支持（灰度在 MedicalAdapter 里会 repeat 到 3通道）
        """
        img = Image.open(path).convert("RGB")  # X-ray 可能是灰度，这里转为 RGB，后续适配器会再规范化
        img = img.resize((self.resize, self.resize))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
        arr = np.transpose(arr, (2,0,1))                 # [3,H,W]
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image"])
        img = self._load_image(img_path)

        if self.binary:
            label = torch.tensor(int(row["label"]), dtype=torch.long)
        else:
            label_vals = [int(row[c]) for c in self.label_cols]
            label = torch.tensor(label_vals, dtype=torch.float32)  # 多标签

        return AdapterInput(
            raw_data=img,   # MedicalAdapter 会处理 [1, H, W] or [3, H, W]
            labels=label,
            metadata={"domain": "medical"}
        )


# ---------------------------
# 统一构建 DataLoader
# ---------------------------

def _split_train_val(ds: Dataset, val_ratio: float = 0.1, seed: int = 42):
    if len(ds) < 2:
        return ds, None
    val_len = max(1, int(len(ds) * val_ratio))
    train_len = len(ds) - val_len
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [train_len, val_len], generator=gen)


def _make_loader(ds: Dataset, cfg: LoaderConfig, train: bool):
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(cfg.shuffle_train and train),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False
    )


def build_all_dataloaders(cfg: LoaderConfig) -> Dict[str, Dict[str, DataLoader]]:
    """
    返回：
    {
      'vision':   {'train': dl, 'val': dl2},
      'nlp':      {'train': dl, 'val': dl2},
      'security': {'train': dl, 'val': dl2},
      'medical':  {'train': dl, 'val': dl2},
    }
    其中可能某些 'val' 为 None（如果数据太少或未提供验证集）
    """
    out: Dict[str, Dict[str, DataLoader]] = {}

    # Vision
    v_train = VisionCIFAR10Dataset(cfg, train=True)
    v_val   = VisionCIFAR10Dataset(cfg, train=False)
    out["vision"] = {
        "train": _make_loader(v_train, cfg, True),
        "val":   _make_loader(v_val, cfg, False)
    }

    # NLP
    nlp_train = NLPTextCSVDataset(cfg.nlp_csv_path, lower=cfg.nlp_lower)
    if cfg.nlp_val_csv_path and os.path.isfile(cfg.nlp_val_csv_path):
        nlp_val = NLPTextCSVDataset(cfg.nlp_val_csv_path, lower=cfg.nlp_lower)
    else:
        nlp_train, nlp_val = _split_train_val(nlp_train)
    out["nlp"] = {
        "train": _make_loader(nlp_train, cfg, True),
        "val":   None if nlp_val is None else _make_loader(nlp_val, cfg, False)
    }

    # Security
    sec_train = SecurityNSLKDDDataset(
        cfg.sec_csv_path, cfg.sec_label_col, cfg.sec_ignore_cols, cfg.sec_limit
    )
    sec_train, sec_val = _split_train_val(sec_train)
    out["security"] = {
        "train": _make_loader(sec_train, cfg, True),
        "val":   None if sec_val is None else _make_loader(sec_val, cfg, False)
    }

    # Medical
    med_train = MedicalXrayCSVDataset(
        image_root=cfg.med_image_root,
        label_csv=cfg.med_label_csv,
        resize=cfg.med_resize,
        binary=cfg.med_binary,
        label_cols=cfg.med_label_cols
    )
    med_train, med_val = _split_train_val(med_train)
    out["medical"] = {
        "train": _make_loader(med_train, cfg, True),
        "val":   None if med_val is None else _make_loader(med_val, cfg, False)
    }

    return out


# ---------------------------
# 快速自检（可直接运行）
# ---------------------------

if __name__ == "__main__":
    cfg = LoaderConfig(
        batch_size=8,
        num_workers=0,  # Windows/IDE 稳妥
        # 下面这些路径按需改成你的实际位置；如果先不准备 Medical/NLP/Sec 数据，
        # 也可以只跑 Vision（CIFAR-10 会自动下载）
        nlp_csv_path="./datasets/imdb/train.csv",  # 示例：text,label
        nlp_val_csv_path=None,
        sec_csv_path="./datasets/nsl_kdd/nsl_kdd.csv",
        med_image_root="./datasets/chestxray14/images",
        med_label_csv="./datasets/chestxray14/labels.csv",
        med_binary=True,           # =True 时 labels.csv 两列: image,label
        med_label_cols=None        # 多标签时才填写，如 ["Atelectasis","Cardiomegaly",...]
    )

    dls = build_all_dataloaders(cfg)

    print("\n✅ DataLoaders ready:")
    for dom, pair in dls.items():
        for split, dl in pair.items():
            if dl is None:
                print(f"- {dom}/{split}: None")
            else:
                try:
                    batch: List[AdapterInput] = next(iter(dl))
                    # 这里 batch 是 list? 不是，我们 Dataset 已经返回 AdapterInput，
                    # DataLoader 默认会把它们组成 list。我们只简单打印长度。
                    print(f"- {dom}/{split}: batch_size={len(batch)} ok")
                except StopIteration:
                    print(f"- {dom}/{split}: empty")
                except Exception as e:
                    print(f"- {dom}/{split}: error -> {e}")
