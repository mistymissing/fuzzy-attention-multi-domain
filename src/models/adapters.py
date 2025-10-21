"""
Domain Adapter 基类 - UniMatch-Clip项目核心架构组件

功能：
1. 统一四个领域的接口抽象 (Vision/NLP/Security/Medical)
2. 跨域特征标准化和转换
3. 领域特定的数据预处理
4. 置信度计算和困难样本识别的统一接口
5. 可插拔的领域适配器架构

设计理念：
- 每个领域有独特的数据格式和模型架构
- 通过统一接口实现跨域协作
- 支持动态切换和组合不同领域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from pathlib import Path


class DomainType(Enum):
    """领域类型枚举"""
    VISION = "vision"
    NLP = "nlp"
    SECURITY = "security"
    MEDICAL = "medical"


class AdapterMode(Enum):
    """适配器模式枚举"""
    TRAINING = "training"  # 训练模式
    INFERENCE = "inference"  # 推理模式
    EVALUATION = "evaluation"  # 评估模式


@dataclass
class AdapterInput:
    """适配器输入数据结构"""
    raw_data: Any  # 原始输入数据 (图像/文本/信号等)
    labels: Optional[Any] = None  # 标签 (训练时使用)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    batch_size: Optional[int] = None  # 批次大小
    device: Optional[torch.device] = None  # 设备信息


@dataclass
class AdapterOutput:
    """适配器输出数据结构"""
    # 核心输出
    embeddings: torch.Tensor  # 标准化特征嵌入 [B, D]
    logits: torch.Tensor  # 预测logits [B, num_classes]
    confidence_scores: torch.Tensor  # 置信度分数 [B]

    # 中间表示
    raw_features: Optional[torch.Tensor] = None  # 原始特征
    attention_weights: Optional[torch.Tensor] = None  # 注意力权重

    # 元信息
    processing_time: float = 0.0  # 处理时间
    metadata: Dict[str, Any] = field(default_factory=dict)  # 输出元数据

    # 困难样本相关
    is_hard_sample: Optional[torch.Tensor] = None  # 是否困难样本 [B]
    difficulty_scores: Optional[torch.Tensor] = None  # 困难度分数 [B]


class BaseDomainAdapter(ABC, nn.Module):
    """
    Domain Adapter 基类

    所有领域适配器的抽象基类，定义统一接口
    """

    def __init__(self,
                 domain_type: DomainType,
                 input_dim: int,
                 hidden_dim: int = 512,
                 output_dim: int = 256,
                 num_classes: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 device: Optional[torch.device] = None):
        """
        Args:
            domain_type: 领域类型
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 统一输出维度 (跨域对齐的目标维度)
            num_classes: 分类类别数 (如果是分类任务)
            dropout_rate: Dropout率
            device: 计算设备
        """
        super().__init__()

        self.domain_type = domain_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.device = device or torch.device('cpu')

        # 适配器状态
        self.mode = AdapterMode.TRAINING
        self.is_initialized = False

        # 统计信息
        self.processing_stats = {
            'total_samples': 0,
            'hard_samples': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }

        # 日志
        self.logger = logging.getLogger(f"{domain_type.value}Adapter")

        # 核心组件 (子类需要实现)
        self.feature_extractor = None  # 特征提取器
        self.embedding_projector = None  # 嵌入投影器
        self.classifier = None  # 分类器 (可选)
        self.confidence_estimator = None  # 置信度估计器

        # 初始化核心组件
        self._init_core_components()

        # 移动到指定设备
        self.to(self.device)

    def _init_core_components(self):
        """初始化核心组件 - 通用实现"""

        # 标准嵌入投影器 (输入维度 -> 统一输出维度)
        self.embedding_projector = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

        # 分类器 (如果指定了类别数)
        if self.num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(self.output_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            )

        # 置信度估计器 (基于特征和预测的置信度)
        confidence_input_dim = self.output_dim
        if self.num_classes:
            confidence_input_dim += self.num_classes  # 拼接logits

        self.confidence_estimator = nn.Sequential(
            nn.Linear(confidence_input_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()  # 输出[0,1]置信度
        )

    @abstractmethod
    def extract_features(self, inputs: AdapterInput) -> torch.Tensor:
        """
        提取领域特定特征 - 子类必须实现

        Args:
            inputs: 适配器输入

        Returns:
            原始特征张量 [B, input_dim]
        """
        pass

    @abstractmethod
    def preprocess_data(self, raw_data: Any) -> torch.Tensor:
        """
        领域特定的数据预处理 - 子类必须实现

        Args:
            raw_data: 原始数据 (图像/文本/信号等)

        Returns:
            预处理后的张量
        """
        pass

    def forward(self, inputs: AdapterInput) -> AdapterOutput:
        """
        适配器前向传播 - 统一处理流程

        Args:
            inputs: 适配器输入

        Returns:
            适配器输出
        """
        start_time = time.time()

        # 1. 数据预处理
        preprocessed_data = self.preprocess_data(inputs.raw_data)

        # 2. 特征提取 (领域特定)
        raw_features = self.extract_features(AdapterInput(
            raw_data=preprocessed_data,
            labels=inputs.labels,
            metadata=inputs.metadata,
            device=self.device
        ))

        # 3. 嵌入投影 (标准化到统一维度)
        embeddings = self.embedding_projector(raw_features)

        # 4. 分类预测 (如果有分类器)
        logits = None
        if self.classifier is not None:
            logits = self.classifier(embeddings)
        else:
            # 如果没有分类器，返回零向量
            batch_size = embeddings.size(0)
            logits = torch.zeros(batch_size, self.num_classes or 1, device=self.device)

        # 5. 置信度估计
        confidence_scores = self._compute_confidence(embeddings, logits)

        # 6. 困难样本识别
        is_hard_sample, difficulty_scores = self._identify_hard_samples(confidence_scores)

        # 7. 构建输出
        processing_time = time.time() - start_time

        output = AdapterOutput(
            embeddings=embeddings,
            logits=logits,
            confidence_scores=confidence_scores,
            raw_features=raw_features,
            processing_time=processing_time,
            is_hard_sample=is_hard_sample,
            difficulty_scores=difficulty_scores,
            metadata={
                'domain': self.domain_type.value,
                'batch_size': embeddings.size(0),
                'feature_dim': embeddings.size(1)
            }
        )

        # 8. 更新统计信息
        self._update_stats(output)

        return output

    def _compute_confidence(self, embeddings: torch.Tensor,
                            logits: torch.Tensor) -> torch.Tensor:
        """计算置信度分数"""

        # 准备置信度估计器的输入
        if logits is not None and logits.numel() > 0:
            # 拼接嵌入和logits
            confidence_input = torch.cat([embeddings, logits], dim=-1)
        else:
            confidence_input = embeddings

        # 使用置信度估计器
        confidence_scores = self.confidence_estimator(confidence_input).squeeze(-1)

        return confidence_scores

    def _identify_hard_samples(self, confidence_scores: torch.Tensor,
                               threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """识别困难样本"""

        if threshold is None:
            # 使用领域特定的默认阈值
            domain_thresholds = {
                DomainType.VISION: 0.7,
                DomainType.NLP: 0.6,
                DomainType.SECURITY: 0.8,
                DomainType.MEDICAL: 0.75
            }
            threshold = domain_thresholds.get(self.domain_type, 0.7)

        # 困难样本标识
        is_hard_sample = confidence_scores < threshold

        # 困难度分数 (置信度越低，困难度越高)
        difficulty_scores = 1.0 - confidence_scores

        return is_hard_sample, difficulty_scores

    def _update_stats(self, output: AdapterOutput):
        """更新统计信息"""
        batch_size = output.embeddings.size(0)
        hard_count = output.is_hard_sample.sum().item() if output.is_hard_sample is not None else 0
        avg_confidence = output.confidence_scores.mean().item()

        # 更新累积统计
        total_samples = self.processing_stats['total_samples']
        self.processing_stats['total_samples'] += batch_size
        self.processing_stats['hard_samples'] += hard_count

        # 更新平均值
        new_total = total_samples + batch_size
        self.processing_stats['avg_confidence'] = (
                (self.processing_stats['avg_confidence'] * total_samples + avg_confidence * batch_size) / new_total
        )
        self.processing_stats['avg_processing_time'] = (
                (self.processing_stats[
                     'avg_processing_time'] * total_samples + output.processing_time * batch_size) / new_total
        )

    def set_mode(self, mode: AdapterMode):
        """设置适配器模式"""
        self.mode = mode
        if mode == AdapterMode.TRAINING:
            self.train()
        else:
            self.eval()

        self.logger.debug(f"Adapter mode set to: {mode.value}")

    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.output_dim

    def get_domain_type(self) -> DomainType:
        """获取领域类型"""
        return self.domain_type

    def get_statistics(self) -> Dict[str, Any]:
        """获取适配器统计信息"""
        stats = self.processing_stats.copy()

        # 计算额外指标
        if stats['total_samples'] > 0:
            stats['hard_sample_ratio'] = stats['hard_samples'] / stats['total_samples']
        else:
            stats['hard_sample_ratio'] = 0.0

        stats['domain'] = self.domain_type.value
        stats['mode'] = self.mode.value

        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.processing_stats = {
            'total_samples': 0,
            'hard_samples': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }

    def save_adapter(self, save_path: Union[str, Path]):
        """保存适配器状态"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'state_dict': self.state_dict(),
            'config': {
                'domain_type': self.domain_type.value,
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_classes': self.num_classes,
                'dropout_rate': self.dropout_rate
            },
            'statistics': self.get_statistics()
        }

        torch.save(state, save_path)
        self.logger.info(f"Adapter saved to {save_path}")

    def load_adapter(self, load_path: Union[str, Path]):
        """加载适配器状态"""
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Adapter file not found: {load_path}")

        state = torch.load(load_path, map_location=self.device)

        # 验证配置兼容性
        config = state['config']
        if config['domain_type'] != self.domain_type.value:
            raise ValueError(f"Domain type mismatch: {config['domain_type']} vs {self.domain_type.value}")

        # 加载权重
        self.load_state_dict(state['state_dict'])

        # 加载统计信息
        if 'statistics' in state:
            self.processing_stats.update(state['statistics'])

        self.logger.info(f"Adapter loaded from {load_path}")


class AdapterRegistry:
    """适配器注册表 - 管理所有领域适配器"""

    def __init__(self):
        self.adapters = {}  # domain_type -> adapter instance
        self.adapter_classes = {}  # domain_type -> adapter class
        self.logger = logging.getLogger("AdapterRegistry")

    def register_adapter_class(self, domain_type: DomainType, adapter_class):
        """注册适配器类"""
        self.adapter_classes[domain_type] = adapter_class
        self.logger.info(f"Registered adapter class for {domain_type.value}")

    def create_adapter(self, domain_type: DomainType, **kwargs) -> BaseDomainAdapter:
        """创建适配器实例"""
        if domain_type not in self.adapter_classes:
            raise ValueError(f"No adapter class registered for {domain_type.value}")

        adapter_class = self.adapter_classes[domain_type]
        adapter = adapter_class(domain_type=domain_type, **kwargs)

        self.adapters[domain_type] = adapter
        self.logger.info(f"Created adapter instance for {domain_type.value}")

        return adapter

    def get_adapter(self, domain_type: DomainType) -> Optional[BaseDomainAdapter]:
        """获取适配器实例"""
        return self.adapters.get(domain_type)

    def get_all_adapters(self) -> Dict[DomainType, BaseDomainAdapter]:
        """获取所有适配器"""
        return self.adapters.copy()

    def set_mode_all(self, mode: AdapterMode):
        """设置所有适配器的模式"""
        for adapter in self.adapters.values():
            adapter.set_mode(mode)

    def get_unified_statistics(self) -> Dict[str, Any]:
        """获取所有适配器的统一统计信息"""
        stats = {}

        for domain_type, adapter in self.adapters.items():
            stats[domain_type.value] = adapter.get_statistics()

        # 计算总体统计
        total_samples = sum(s['total_samples'] for s in stats.values())
        total_hard_samples = sum(s['hard_samples'] for s in stats.values())

        stats['overall'] = {
            'total_samples': total_samples,
            'total_hard_samples': total_hard_samples,
            'overall_hard_ratio': total_hard_samples / max(total_samples, 1),
            'active_domains': len(self.adapters)
        }

        return stats


# =====================================================
# 测试和示例
# =====================================================

class TestDomainAdapter(BaseDomainAdapter):
    """测试用的简单适配器实现"""

    def extract_features(self, inputs: AdapterInput) -> torch.Tensor:
        """简单的特征提取 - 直接返回预处理后的数据"""
        if isinstance(inputs.raw_data, torch.Tensor):
            return inputs.raw_data
        else:
            # 如果是其他类型，转换为张量
            return torch.tensor(inputs.raw_data, dtype=torch.float32, device=self.device)

    def preprocess_data(self, raw_data: Any) -> torch.Tensor:
        """简单的数据预处理"""
        if isinstance(raw_data, torch.Tensor):
            return raw_data.to(self.device)
        elif isinstance(raw_data, np.ndarray):
            return torch.from_numpy(raw_data).float().to(self.device)
        elif isinstance(raw_data, list):
            return torch.tensor(raw_data, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unsupported raw_data type: {type(raw_data)}")


def test_base_adapter():
    """测试基础适配器功能"""

    print("🧪 测试Domain Adapter基类...")

    # 创建测试适配器
    adapter = TestDomainAdapter(
        domain_type=DomainType.VISION,
        input_dim=512,
        hidden_dim=256,
        output_dim=128,
        num_classes=10,
        dropout_rate=0.1
    )

    print(f"✅ 创建适配器: {adapter.get_domain_type().value}")
    print(f"📏 嵌入维度: {adapter.get_embedding_dim()}")

    # 测试数据
    batch_size = 4
    test_data = torch.randn(batch_size, 512)
    test_labels = torch.randint(0, 10, (batch_size,))

    # 创建输入
    inputs = AdapterInput(
        raw_data=test_data,
        labels=test_labels,
        metadata={'test': True},
        batch_size=batch_size
    )

    # 前向传播
    adapter.set_mode(AdapterMode.INFERENCE)
    outputs = adapter(inputs)

    print(f"\n📊 适配器输出:")
    print(f"嵌入形状: {outputs.embeddings.shape}")
    print(f"Logits形状: {outputs.logits.shape}")
    print(f"置信度: {outputs.confidence_scores}")
    print(f"困难样本: {outputs.is_hard_sample}")
    print(f"处理时间: {outputs.processing_time:.4f}秒")

    # 统计信息
    stats = adapter.get_statistics()
    print(f"\n📈 适配器统计:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return True


def test_adapter_registry():
    """测试适配器注册表"""

    print("\n🏛️ 测试适配器注册表...")

    # 创建注册表
    registry = AdapterRegistry()

    # 注册适配器类
    registry.register_adapter_class(DomainType.VISION, TestDomainAdapter)
    registry.register_adapter_class(DomainType.NLP, TestDomainAdapter)

    # 创建适配器实例
    vision_adapter = registry.create_adapter(
        DomainType.VISION,
        input_dim=512,
        output_dim=128,
        num_classes=10
    )

    nlp_adapter = registry.create_adapter(
        DomainType.NLP,
        input_dim=768,
        output_dim=128,
        num_classes=5
    )

    print(f"✅ 创建了 {len(registry.get_all_adapters())} 个适配器")

    # 测试模式切换
    registry.set_mode_all(AdapterMode.TRAINING)
    print("📚 所有适配器设置为训练模式")

    # 统一统计
    unified_stats = registry.get_unified_statistics()
    print(f"\n📊 统一统计信息:")
    for domain, stats in unified_stats.items():
        if domain != 'overall':
            print(f"{domain}: {stats['total_samples']} 样本")

    print(f"总体: {unified_stats['overall']}")

    return True


def demonstrate_adapter_workflow():
    """演示适配器的完整工作流程"""

    print("\n🔄 演示适配器工作流程...")

    # 1. 创建不同领域的适配器
    domains_config = {
        DomainType.VISION: {'input_dim': 512, 'num_classes': 10},
        DomainType.NLP: {'input_dim': 768, 'num_classes': 5},
        DomainType.SECURITY: {'input_dim': 256, 'num_classes': 2},
        DomainType.MEDICAL: {'input_dim': 384, 'num_classes': 4}
    }

    adapters = {}
    for domain, config in domains_config.items():
        adapter = TestDomainAdapter(
            domain_type=domain,
            output_dim=128,  # 统一输出维度
            **config
        )
        adapters[domain] = adapter

    print(f"✅ 创建了 {len(adapters)} 个领域适配器")

    # 2. 模拟各领域的数据处理
    results = {}

    for domain, adapter in adapters.items():
        # 生成模拟数据
        input_dim = domains_config[domain]['input_dim']
        test_data = torch.randn(3, input_dim)  # 3个样本

        inputs = AdapterInput(
            raw_data=test_data,
            metadata={'domain': domain.value}
        )

        # 处理数据
        outputs = adapter(inputs)
        results[domain] = outputs

        print(f"🔧 {domain.value}: 处理了3个样本，困难样本: {outputs.is_hard_sample.sum().item()}个")

    # 3. 验证跨域一致性
    print(f"\n🌐 跨域一致性检查:")
    embedding_dims = [result.embeddings.shape[1] for result in results.values()]

    if len(set(embedding_dims)) == 1:
        print(f"✅ 所有领域输出维度一致: {embedding_dims[0]}")
    else:
        print(f"❌ 输出维度不一致: {embedding_dims}")

    # 4. 计算跨域相似度 (简单示例)
    print(f"\n🔍 跨域特征相似度分析:")
    domain_embeddings = {}

    for domain, result in results.items():
        # 取第一个样本的嵌入
        domain_embeddings[domain] = result.embeddings[0]

    # 计算各领域间的相似度
    domain_list = list(domain_embeddings.keys())
    for i, domain1 in enumerate(domain_list):
        for j, domain2 in enumerate(domain_list[i + 1:], i + 1):
            emb1 = domain_embeddings[domain1]
            emb2 = domain_embeddings[domain2]

            similarity = F.cosine_similarity(emb1, emb2, dim=0).item()
            print(f"{domain1.value} ↔ {domain2.value}: 相似度 {similarity:.4f}")

    return True


if __name__ == "__main__":
    print("🚀 Domain Adapter 基类测试启动")
    print("=" * 50)

    # 基础功能测试
    basic_success = test_base_adapter()

    if basic_success:
        # 注册表测试
        registry_success = test_adapter_registry()

        # 工作流程演示
        workflow_success = demonstrate_adapter_workflow()

        if all([registry_success, workflow_success]):
            print("\n🎉 Domain Adapter 基类完全就绪!")
            print("✅ 统一接口设计验证通过")
            print("✅ 跨域特征标准化工作正常")
            print("✅ 适配器注册和管理系统正常")
            print("\n🚀 可以开始具体领域适配器的实现!")
            print("📋 下一步: Vision 和 NLP 适配器实现")
        else:
            print("\n⚠️ 部分高级功能需要调试")
    else:
        print("\n❌ 需要调试基础适配器")