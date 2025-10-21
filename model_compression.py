#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniMatch-Clip 第五天：模型压缩与优化
实现知识蒸馏、量化优化和移动端适配等模型压缩技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
from torch.nn.utils import prune
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import logging
from abc import ABC, abstractmethod
import copy
import warnings

# 忽略量化警告
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class CompressionConfig:
    """压缩配置"""
    # 知识蒸馏
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    student_teacher_ratio: float = 0.5  # 学生模型相对于教师模型的大小比例

    # 量化
    quantization_backend: str = 'fbgemm'  # 'fbgemm' for x86, 'qnnpack' for ARM
    quantization_dtype: torch.dtype = torch.qint8
    enable_dynamic_quantization: bool = True
    enable_static_quantization: bool = True

    # 剪枝
    pruning_ratio: float = 0.3
    structured_pruning: bool = True

    # 其他优化
    enable_jit_compilation: bool = True
    enable_half_precision: bool = True
    batch_size_optimization: bool = True

class KnowledgeDistiller:
    """知识蒸馏器"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = logging.getLogger('KnowledgeDistiller')

    def create_student_model(self, teacher_model: nn.Module, model_type: str) -> nn.Module:
        """创建学生模型（简化版的教师模型）"""
        if model_type == 'vision':
            return self._create_vision_student(teacher_model)
        elif model_type == 'nlp':
            return self._create_nlp_student(teacher_model)
        elif model_type == 'security':
            return self._create_security_student(teacher_model)
        elif model_type == 'medical':
            return self._create_medical_student(teacher_model)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_vision_student(self, teacher: nn.Module) -> nn.Module:
        """创建视觉学生模型"""
        class CompactVisionModel(nn.Module):
            def __init__(self, output_dim=128):
                super().__init__()
                # 更简单的CNN结构
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),  # 减少通道数
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(32 * 4 * 4, 64),  # 减少隐藏层大小
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.classifier = nn.Linear(output_dim, 10)  # 假设10类

            def forward(self, x):
                embeddings = self.backbone(x)
                logits = self.classifier(embeddings)
                return {
                    'embeddings': embeddings,
                    'logits': logits,
                    'confidence_scores': F.softmax(logits, dim=-1).max(dim=-1)[0]
                }

        return CompactVisionModel()

    def _create_nlp_student(self, teacher: nn.Module) -> nn.Module:
        """创建NLP学生模型"""
        class CompactNLPModel(nn.Module):
            def __init__(self, output_dim=128, vocab_size=1000):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 32)  # 减少嵌入维度
                self.encoder = nn.Sequential(
                    nn.Linear(32 * 20, 64),  # 假设最大序列长度20
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.classifier = nn.Linear(output_dim, 2)  # 假设2类

            def forward(self, x):
                # 简化的文本处理
                if isinstance(x, str):
                    # 简单的字符级编码
                    chars = [ord(c) % 1000 for c in x[:20]]
                    chars.extend([0] * (20 - len(chars)))  # 填充
                    x = torch.tensor([chars], device=next(self.parameters()).device)

                embedded = self.embedding(x)
                embedded = embedded.view(embedded.size(0), -1)
                embeddings = self.encoder(embedded)
                logits = self.classifier(embeddings)

                return {
                    'embeddings': embeddings,
                    'logits': logits,
                    'confidence_scores': F.softmax(logits, dim=-1).max(dim=-1)[0]
                }

        return CompactNLPModel()

    def _create_security_student(self, teacher: nn.Module) -> nn.Module:
        """创建安全学生模型"""
        class CompactSecurityModel(nn.Module):
            def __init__(self, feature_dim=122, output_dim=128):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(feature_dim, 64),  # 减少隐藏层
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.classifier = nn.Linear(output_dim, 5)  # 假设5类

            def forward(self, x):
                embeddings = self.encoder(x)
                logits = self.classifier(embeddings)
                return {
                    'embeddings': embeddings,
                    'logits': logits,
                    'confidence_scores': F.softmax(logits, dim=-1).max(dim=-1)[0]
                }

        return CompactSecurityModel()

    def _create_medical_student(self, teacher: nn.Module) -> nn.Module:
        """创建医疗学生模型"""
        class CompactMedicalModel(nn.Module):
            def __init__(self, output_dim=128):
                super().__init__()
                # 简化的医疗图像处理
                self.backbone = nn.Sequential(
                    nn.Conv2d(1, 8, 3, padding=1),  # 单通道输入，减少通道数
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(16 * 4 * 4, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.classifier = nn.Linear(output_dim, 14)  # 假设14种疾病

            def forward(self, x):
                embeddings = self.backbone(x)
                logits = self.classifier(embeddings)
                return {
                    'embeddings': embeddings,
                    'logits': logits,
                    'confidence_scores': F.softmax(logits, dim=-1).max(dim=-1)[0]
                }

        return CompactMedicalModel()

    def distill_knowledge(self, teacher: nn.Module, student: nn.Module,
                         train_loader: List[Tuple], num_epochs: int = 5) -> nn.Module:
        """执行知识蒸馏"""
        self.logger.info("开始知识蒸馏训练...")

        teacher.eval()
        student.train()

        optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
        criterion = nn.KLDivLoss(reduction='batchmean')

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch_data in train_loader[:10]:  # 限制batch数量用于演示
                inputs, _ = batch_data if isinstance(batch_data, tuple) else (batch_data, None)

                # 教师模型预测
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                    teacher_logits = teacher_outputs.get('logits', torch.randn(inputs.shape[0], 10))

                # 学生模型预测
                student_outputs = student(inputs)
                student_logits = student_outputs.get('logits', torch.randn(inputs.shape[0], 10))

                # 蒸馏损失
                T = self.config.distillation_temperature
                soft_targets = F.softmax(teacher_logits / T, dim=-1)
                soft_prob = F.log_softmax(student_logits / T, dim=-1)

                distill_loss = criterion(soft_prob, soft_targets) * (T * T)

                # 总损失
                total_loss = distill_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss += total_loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")

        self.logger.info("知识蒸馏完成")
        return student

class ModelQuantizer:
    """模型量化器"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = logging.getLogger('ModelQuantizer')

    def dynamic_quantize(self, model: nn.Module) -> nn.Module:
        """动态量化"""
        self.logger.info("执行动态量化...")

        # 设置量化后端
        if self.config.quantization_backend == 'fbgemm':
            torch.backends.quantized.engine = 'fbgemm'
        elif self.config.quantization_backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'

        # 动态量化只量化权重
        quantized_model = quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # 要量化的层类型
            dtype=self.config.quantization_dtype
        )

        self.logger.info("动态量化完成")
        return quantized_model

    def static_quantize(self, model: nn.Module, calibration_loader: List) -> nn.Module:
        """静态量化（需要校准数据）"""
        self.logger.info("执行静态量化...")

        # 准备量化
        model.eval()
        model.qconfig = quantization.get_default_qconfig(self.config.quantization_backend)
        model_prepared = quantization.prepare(model)

        # 校准
        self.logger.info("开始校准...")
        with torch.no_grad():
            for i, data in enumerate(calibration_loader[:10]):  # 限制校准数据量
                if isinstance(data, tuple):
                    data = data[0]
                try:
                    _ = model_prepared(data)
                except Exception as e:
                    self.logger.warning(f"校准步骤 {i} 失败: {e}")

        # 转换为量化模型
        quantized_model = quantization.convert(model_prepared)

        self.logger.info("静态量化完成")
        return quantized_model

class ModelPruner:
    """模型剪枝器"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = logging.getLogger('ModelPruner')

    def unstructured_prune(self, model: nn.Module) -> nn.Module:
        """非结构化剪枝"""
        self.logger.info(f"执行非结构化剪枝，剪枝比例: {self.config.pruning_ratio}")

        # 收集要剪枝的参数
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        # 全局非结构化剪枝
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_ratio,
        )

        # 移除剪枝重参数化
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        self.logger.info("非结构化剪枝完成")
        return model

    def structured_prune(self, model: nn.Module) -> nn.Module:
        """结构化剪枝"""
        self.logger.info(f"执行结构化剪枝，剪枝比例: {self.config.pruning_ratio}")

        # 对卷积层进行结构化剪枝
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=self.config.pruning_ratio,
                    n=2,
                    dim=0
                )
                prune.remove(module, 'weight')
            elif isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                # 对线性层进行非结构化剪枝
                prune.l1_unstructured(module, name="weight", amount=self.config.pruning_ratio)
                prune.remove(module, 'weight')

        self.logger.info("结构化剪枝完成")
        return model

class ModelOptimizer:
    """模型优化器"""

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = logging.getLogger('ModelOptimizer')

    def optimize_inference(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """推理优化"""
        optimizations = []

        # JIT编译
        if self.config.enable_jit_compilation:
            try:
                model = torch.jit.trace(model, sample_input)
                optimizations.append("JIT编译")
            except Exception as e:
                self.logger.warning(f"JIT编译失败: {e}")

        # 半精度
        if self.config.enable_half_precision and torch.cuda.is_available():
            try:
                model = model.half()
                optimizations.append("半精度")
            except Exception as e:
                self.logger.warning(f"半精度转换失败: {e}")

        # 融合操作
        try:
            if hasattr(torch.nn.utils, 'fuse_conv_bn_eval'):
                model = torch.nn.utils.fuse_conv_bn_eval(model)
                optimizations.append("卷积BN融合")
        except Exception as e:
            self.logger.warning(f"操作融合失败: {e}")

        self.logger.info(f"应用的优化: {', '.join(optimizations)}")
        return model

class CompressionPipeline:
    """压缩流水线"""

    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.distiller = KnowledgeDistiller(self.config)
        self.quantizer = ModelQuantizer(self.config)
        self.pruner = ModelPruner(self.config)
        self.optimizer = ModelOptimizer(self.config)
        self.logger = logging.getLogger('CompressionPipeline')

    def compress_model(self, model: nn.Module, model_type: str,
                      calibration_data: List = None) -> Dict[str, Any]:
        """完整的模型压缩流程"""
        results = {}

        self.logger.info(f"开始压缩 {model_type} 模型...")

        # 记录原始模型大小
        original_size = self._get_model_size(model)
        results['original_size'] = original_size

        # 1. 知识蒸馏（创建学生模型）
        if self.config.student_teacher_ratio < 1.0:
            try:
                student_model = self.distiller.create_student_model(model, model_type)

                # 模拟训练数据用于蒸馏
                if calibration_data:
                    train_data = calibration_data
                else:
                    train_data = self._generate_dummy_data(model_type, 10)

                student_model = self.distiller.distill_knowledge(model, student_model, train_data)
                model = student_model
                results['distillation_applied'] = True
            except Exception as e:
                self.logger.warning(f"知识蒸馏失败: {e}")
                results['distillation_applied'] = False

        # 2. 剪枝
        try:
            if self.config.structured_pruning:
                model = self.pruner.structured_prune(copy.deepcopy(model))
            else:
                model = self.pruner.unstructured_prune(copy.deepcopy(model))
            results['pruning_applied'] = True
        except Exception as e:
            self.logger.warning(f"剪枝失败: {e}")
            results['pruning_applied'] = False

        # 3. 量化
        compressed_models = {}

        # 动态量化
        if self.config.enable_dynamic_quantization:
            try:
                dynamic_model = self.quantizer.dynamic_quantize(copy.deepcopy(model))
                compressed_models['dynamic_quantized'] = dynamic_model
                results['dynamic_quantization_applied'] = True
            except Exception as e:
                self.logger.warning(f"动态量化失败: {e}")
                results['dynamic_quantization_applied'] = False

        # 静态量化
        if self.config.enable_static_quantization and calibration_data:
            try:
                static_model = self.quantizer.static_quantize(copy.deepcopy(model), calibration_data)
                compressed_models['static_quantized'] = static_model
                results['static_quantization_applied'] = True
            except Exception as e:
                self.logger.warning(f"静态量化失败: {e}")
                results['static_quantization_applied'] = False

        # 4. 推理优化
        sample_input = self._generate_sample_input(model_type)
        for name, compressed_model in compressed_models.items():
            try:
                optimized_model = self.optimizer.optimize_inference(compressed_model, sample_input)
                compressed_models[name] = optimized_model
            except Exception as e:
                self.logger.warning(f"{name} 推理优化失败: {e}")

        # 如果没有量化模型，至少优化原始模型
        if not compressed_models:
            try:
                optimized_model = self.optimizer.optimize_inference(copy.deepcopy(model), sample_input)
                compressed_models['optimized'] = optimized_model
            except Exception as e:
                self.logger.warning(f"推理优化失败: {e}")
                compressed_models['original'] = model

        # 5. 评估压缩效果
        for name, compressed_model in compressed_models.items():
            compressed_size = self._get_model_size(compressed_model)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            results[f'{name}_size'] = compressed_size
            results[f'{name}_compression_ratio'] = compression_ratio

        results['compressed_models'] = compressed_models
        results['best_model'] = self._select_best_model(compressed_models, results)

        self.logger.info(f"模型压缩完成，最佳压缩比: {results.get('best_compression_ratio', 1.0):.2f}x")

        return results

    def _get_model_size(self, model: nn.Module) -> float:
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def _generate_dummy_data(self, model_type: str, batch_size: int = 4) -> List:
        """生成用于测试的虚拟数据"""
        dummy_data = []

        for _ in range(batch_size):
            if model_type == 'vision':
                data = torch.randn(1, 3, 224, 224)
            elif model_type == 'nlp':
                data = "sample text"
            elif model_type == 'security':
                data = torch.randn(1, 122)
            elif model_type == 'medical':
                data = torch.randn(1, 1, 224, 224)
            else:
                data = torch.randn(1, 128)

            dummy_data.append((data, None))

        return dummy_data

    def _generate_sample_input(self, model_type: str) -> torch.Tensor:
        """生成样本输入用于JIT追踪"""
        if model_type == 'vision':
            return torch.randn(1, 3, 224, 224)
        elif model_type == 'nlp':
            return torch.randint(0, 1000, (1, 20))  # 假设词汇表大小1000，序列长度20
        elif model_type == 'security':
            return torch.randn(1, 122)
        elif model_type == 'medical':
            return torch.randn(1, 1, 224, 224)
        else:
            return torch.randn(1, 128)

    def _select_best_model(self, models: Dict[str, nn.Module], results: Dict[str, Any]) -> str:
        """选择最佳压缩模型"""
        best_model_name = None
        best_ratio = 0

        for name in models.keys():
            ratio_key = f'{name}_compression_ratio'
            if ratio_key in results and results[ratio_key] > best_ratio:
                best_ratio = results[ratio_key]
                best_model_name = name

        if best_model_name:
            results['best_compression_ratio'] = best_ratio

        return best_model_name or list(models.keys())[0]

# 演示和测试
class CompressionDemo:
    """模型压缩演示"""

    def __init__(self):
        self.config = CompressionConfig(
            pruning_ratio=0.2,
            distillation_temperature=3.0,
            enable_dynamic_quantization=True,
            enable_static_quantization=False  # 简化演示
        )
        self.pipeline = CompressionPipeline(self.config)

    def demo_single_model_compression(self):
        """单模型压缩演示"""
        print("\n=== 单模型压缩演示 ===")

        # 创建一个简单的测试模型
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.fc = nn.Linear(16 * 224 * 224, 128)
                self.classifier = nn.Linear(128, 10)

            def forward(self, x):
                x = F.relu(self.conv(x))
                x = x.view(x.size(0), -1)
                embeddings = F.relu(self.fc(x))
                logits = self.classifier(embeddings)
                return {
                    'embeddings': embeddings,
                    'logits': logits,
                    'confidence_scores': F.softmax(logits, dim=-1).max(dim=-1)[0]
                }

        model = TestModel()
        model.eval()

        print(f"原始模型大小: {self.pipeline._get_model_size(model):.2f} MB")

        # 压缩模型
        results = self.pipeline.compress_model(model, 'vision')

        print("压缩结果:")
        for key, value in results.items():
            if key.endswith('_size'):
                print(f"  {key}: {value:.2f} MB")
            elif key.endswith('_compression_ratio'):
                print(f"  {key}: {value:.2f}x")
            elif key.endswith('_applied'):
                print(f"  {key}: {value}")

        print(f"最佳模型: {results['best_model']}")
        print(f"最佳压缩比: {results.get('best_compression_ratio', 1.0):.2f}x")

    def demo_multiple_models_compression(self):
        """多模型压缩演示"""
        print("\n=== 多模型压缩演示 ===")

        # 模拟不同类型的模型
        models = {
            'vision': nn.Sequential(
                nn.Conv2d(3, 32, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 128),
                nn.Linear(128, 10)
            ),
            'nlp': nn.Sequential(
                nn.Embedding(1000, 64),
                nn.Flatten(),
                nn.Linear(64 * 20, 128),
                nn.Linear(128, 2)
            ),
            'security': nn.Sequential(
                nn.Linear(122, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.Linear(128, 5)
            )
        }

        compression_summary = {}

        for model_type, model in models.items():
            print(f"\n压缩 {model_type} 模型:")

            # 包装模型以提供兼容的输出格式
            class WrappedModel(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model

                def forward(self, x):
                    if hasattr(self.base_model, '__len__'):
                        # 如果是Sequential模型
                        if model_type == 'nlp' and isinstance(x, str):
                            # 简单的文本编码
                            chars = [ord(c) % 1000 for c in x[:20]]
                            chars.extend([0] * (20 - len(chars)))
                            x = torch.tensor([chars])

                        logits = self.base_model(x)
                    else:
                        logits = self.base_model(x)

                    return {
                        'logits': logits,
                        'embeddings': logits,  # 简化，实际应该是中间层输出
                        'confidence_scores': F.softmax(logits, dim=-1).max(dim=-1)[0]
                    }

            wrapped_model = WrappedModel(model)
            wrapped_model.eval()

            original_size = self.pipeline._get_model_size(wrapped_model)
            results = self.pipeline.compress_model(wrapped_model, model_type)

            best_ratio = results.get('best_compression_ratio', 1.0)
            compression_summary[model_type] = {
                'original_size': original_size,
                'best_compression_ratio': best_ratio,
                'compressed_size': original_size / best_ratio
            }

            print(f"  原始大小: {original_size:.2f} MB")
            print(f"  压缩比: {best_ratio:.2f}x")
            print(f"  压缩后大小: {original_size / best_ratio:.2f} MB")

        # 总结
        print(f"\n=== 压缩总结 ===")
        total_original = sum([info['original_size'] for info in compression_summary.values()])
        total_compressed = sum([info['compressed_size'] for info in compression_summary.values()])
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 1.0

        print(f"总原始大小: {total_original:.2f} MB")
        print(f"总压缩后大小: {total_compressed:.2f} MB")
        print(f"总体压缩比: {overall_ratio:.2f}x")
        print(f"节省空间: {total_original - total_compressed:.2f} MB ({((total_original - total_compressed) / total_original * 100):.1f}%)")

    def run_demo(self):
        """运行完整演示"""
        print("UniMatch-Clip 模型压缩演示")
        print("=" * 50)

        self.demo_single_model_compression()
        self.demo_multiple_models_compression()

        print("\n演示完成!")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 运行演示
    demo = CompressionDemo()
    demo.run_demo()