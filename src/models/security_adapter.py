"""
Security 适配器 - UniMatch-Clip项目安全领域实现

功能：
1. 网络入侵检测
2. 异常流量识别
3. 恶意软件检测
4. 零日攻击识别
5. 安全事件分级

支持的数据集：
- NSL-KDD (入侵检测)
- CICIDS2017 (网络流量)
- UNSW-NB15 (网络攻击)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 导入基类
from base_adapter import BaseDomainAdapter, DomainType, AdapterInput, AdapterOutput, AdapterMode


class SecurityTaskType:
    """安全任务类型"""
    INTRUSION_DETECTION = "intrusion_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    MALWARE_CLASSIFICATION = "malware_classification"
    THREAT_ASSESSMENT = "threat_assessment"


@dataclass
class SecurityFeatures:
    """安全特征数据结构"""
    network_features: Optional[torch.Tensor] = None  # 网络流量特征
    packet_features: Optional[torch.Tensor] = None  # 数据包特征
    temporal_features: Optional[torch.Tensor] = None  # 时序特征
    statistical_features: Optional[torch.Tensor] = None  # 统计特征


class SecurityFeatureExtractor(nn.Module):
    """安全特征提取器"""

    def __init__(self, input_dim: int = 122, hidden_dim: int = 256):
        super().__init__()

        # NSL-KDD有122个特征
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 时序建模（可选）
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor, use_temporal: bool = False) -> torch.Tensor:
        """
        提取安全特征

        Args:
            x: [B, D] 或 [B, T, D] 输入特征
            use_temporal: 是否使用时序编码
        """
        if x.dim() == 3 and use_temporal:
            # 时序数据 [B, T, D]
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            features = self.feature_encoder(x_flat)
            features = features.reshape(B, T, -1)

            # LSTM编码
            lstm_out, _ = self.temporal_encoder(features)
            return lstm_out[:, -1, :]  # 取最后时刻
        else:
            # 静态数据 [B, D]
            if x.dim() == 3:
                x = x.mean(dim=1)  # 时序平均
            return self.feature_encoder(x)


class ThreatLevelClassifier(nn.Module):
    """威胁等级分类器"""

    def __init__(self, input_dim: int = 256, num_threat_levels: int = 5):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_threat_levels)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """威胁等级分类"""
        return self.classifier(features)


class SecurityAdapter(BaseDomainAdapter):
    """
    Security领域适配器
    专门处理网络安全数据，支持入侵检测、异常检测等任务
    """

    def __init__(self,
                 task_type: str = SecurityTaskType.INTRUSION_DETECTION,
                 feature_dim: int = 122,  # NSL-KDD默认特征维度
                 hidden_dim: int = 256,
                 num_classes: Optional[int] = None,
                 use_temporal: bool = False,
                 **kwargs):

        self.task_type = task_type
        self.feature_dim = feature_dim
        self.use_temporal = use_temporal

        # 优先从 kwargs 里读取 device；否则自动选 cuda→cpu
        device = kwargs.pop("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 初始化基类（把 device 传下去，基类会把已注册的子模块搬到 device）
        super().__init__(
            domain_type=DomainType.SECURITY,
            input_dim=hidden_dim,
            num_classes=num_classes or 5,  # 默认5类攻击
            hidden_dim=hidden_dim,
            device=device,
            **kwargs
        )

        # —— 下面这些是在 super().__init__ 之后新建的子模块（默认还在 CPU）——
        self.feature_extractor = SecurityFeatureExtractor(
            input_dim=feature_dim,
            hidden_dim=hidden_dim
        )

        self.threat_classifier = ThreatLevelClassifier(
            input_dim=hidden_dim,
            num_threat_levels=5
        )

        # 数据预处理器（非 nn.Module）
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.logger = logging.getLogger("SecurityAdapter")

        # 关键：把新建的 Module 也迁到同一 device（包含 LSTM 的权重）
        self.to(self.device)

    def preprocess_data(self, raw_data: Any) -> torch.Tensor:
        """预处理安全数据：统一到 self.device 且 float32，输出形状 [B, D]"""

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            elif isinstance(x, list):
                return torch.tensor(x)
            elif isinstance(x, pd.DataFrame):
                # 仅取数值列；若 scaler 已拟合则做标准化
                numeric = x.select_dtypes(include=[np.number]).values
                if hasattr(self.scaler, "mean_"):
                    numeric = self.scaler.transform(numeric)
                return torch.from_numpy(numeric)
            elif isinstance(x, dict):
                return self._extract_network_features(x)
            else:
                raise ValueError(f"Unsupported data type: {type(x)}")

        x = to_tensor(raw_data)

        # 至少 2 维：[B, D]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 统一设备 & 精度
        x = x.to(self.device, dtype=torch.float32)

        return x

    def _extract_network_features(self, flow_dict: dict) -> torch.Tensor:
        """从网络流量字典提取特征"""

        # 基础网络特征
        features = []

        # 数据包统计
        features.append(flow_dict.get('packet_count', 0))
        features.append(flow_dict.get('byte_count', 0))
        features.append(flow_dict.get('duration', 0))

        # 协议特征
        features.append(flow_dict.get('protocol', 0))
        features.append(flow_dict.get('src_port', 0))
        features.append(flow_dict.get('dst_port', 0))

        # 标志位
        features.append(flow_dict.get('syn_flag', 0))
        features.append(flow_dict.get('ack_flag', 0))
        features.append(flow_dict.get('fin_flag', 0))

        # 统计特征
        features.append(flow_dict.get('avg_packet_size', 0))
        features.append(flow_dict.get('flow_iat_mean', 0))
        features.append(flow_dict.get('flow_iat_std', 0))

        # 填充到固定维度
        while len(features) < self.feature_dim:
            features.append(0)

        return torch.tensor(features[:self.feature_dim], dtype=torch.float32)

    def extract_features(self, inputs: AdapterInput) -> torch.Tensor:
        """提取安全特征"""

        # 预处理输入
        processed_data = self.preprocess_data(inputs.raw_data)

        # 确保维度正确
        if processed_data.dim() == 1:
            processed_data = processed_data.unsqueeze(0)

        # 特征提取
        with torch.set_grad_enabled(self.training):
            features = self.feature_extractor(processed_data, self.use_temporal)

        return features

    def _compute_confidence(self, embeddings: torch.Tensor,
                            logits: torch.Tensor) -> torch.Tensor:
        """计算安全置信度"""

        # 基础置信度
        base_conf = super()._compute_confidence(embeddings, logits)

        # 安全特定增强
        if logits is not None:
            # 威胁等级置信度
            threat_probs = F.softmax(logits, dim=-1)

            # 正常流量的概率作为置信度的一部分
            if threat_probs.size(-1) >= 1:
                normal_prob = threat_probs[:, 0]  # 假设第0类是正常

                # 异常检测置信度：越偏离正常，置信度越低
                anomaly_conf = 1.0 - torch.abs(normal_prob - 0.5) * 2

                # 组合置信度
                final_conf = base_conf * 0.7 + anomaly_conf * 0.3
            else:
                final_conf = base_conf
        else:
            final_conf = base_conf

        return torch.clamp(final_conf, 0.0, 1.0)

    def detect_zero_day_attack(self, inputs: AdapterInput) -> Dict[str, Any]:
        """检测零日攻击"""

        features = self.extract_features(inputs)  # 256维

        # 异常度计算
        with torch.no_grad():
            # 先投影到低维，再重构回高维
            embedded = self.embedding_projector(features)  # 256->128

            # 创建一个简单的重构网络或直接使用嵌入距离
            # 方案1：使用嵌入空间的异常度
            mean_embedding = embedded.mean(dim=0, keepdim=True)
            anomaly_scores = torch.norm(embedded - mean_embedding, dim=1)

            # 异常阈值
            anomaly_threshold = 2.0
            is_zero_day = anomaly_scores > anomaly_threshold

        return {
            'is_zero_day': is_zero_day.cpu().numpy(),
            'anomaly_score': anomaly_scores.cpu().numpy(),
            'features': features.cpu().numpy()
        }

    def forward(self, inputs: AdapterInput) -> AdapterOutput:
        """前向传播"""

        outputs = super().forward(inputs)

        # 添加安全特定信息
        outputs.metadata.update({
            'task_type': self.task_type,
            'feature_dim': self.feature_dim,
            'use_temporal': self.use_temporal
        })

        # 如果是入侵检测，添加威胁等级
        if self.task_type == SecurityTaskType.INTRUSION_DETECTION:
            # 使用原始的hidden维特征而不是投影后的embeddings
            features = self.extract_features(inputs)  # 这是256维的
            threat_logits = self.threat_classifier(features)
            threat_probs = F.softmax(threat_logits, dim=-1)
            outputs.metadata['threat_levels'] = threat_probs.detach().cpu()

        return outputs


# =====================================================
# 测试函数
# =====================================================

def test_security_adapter():
    """测试Security适配器"""

    print("🔒 测试Security适配器...")

    # 创建适配器
    adapter = SecurityAdapter(
        task_type=SecurityTaskType.INTRUSION_DETECTION,
        feature_dim=122,
        num_classes=5,
        hidden_dim=256,
        output_dim=128
    )

    print(f"✅ 创建Security适配器")
    print(f"📏 输出维度: {adapter.get_embedding_dim()}")

    # 模拟NSL-KDD数据
    batch_size = 4
    test_features = torch.randn(batch_size, 122)  # 122个特征
    test_labels = torch.randint(0, 5, (batch_size,))

    # 创建输入
    inputs = AdapterInput(
        raw_data=test_features,
        labels=test_labels,
        metadata={'source': 'nsl_kdd'}
    )

    # 处理
    adapter.set_mode(AdapterMode.INFERENCE)
    outputs = adapter(inputs)

    print(f"\n📈 处理结果:")
    print(f"嵌入形状: {outputs.embeddings.shape}")
    print(f"置信度: {outputs.confidence_scores.detach().numpy()}")
    print(f"困难样本: {outputs.is_hard_sample.numpy()}")

    if 'threat_levels' in outputs.metadata:
        print(f"威胁等级: {outputs.metadata['threat_levels'].numpy()}")

    # 测试零日攻击检测
    zero_day_result = adapter.detect_zero_day_attack(inputs)
    print(f"\n🎯 零日攻击检测:")
    print(f"异常分数: {zero_day_result['anomaly_score']}")
    print(f"检测结果: {zero_day_result['is_zero_day']}")

    return True


if __name__ == "__main__":
    test_success = test_security_adapter()
    if test_success:
        print("\n✅ Security适配器就绪!")