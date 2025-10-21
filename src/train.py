#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniMatch-Clip 第五天：统一推理框架
实现多域适配器的统一调用接口和批量推理管道
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入适配器
try:
    from vision_adapter import VisionAdapter
    from nlp_adapter import NLPAdapter
    from security_adapter import SecurityAdapter
    from medical_adapter import MedicalAdapter, MedicalModalityType
    from base_adapter import AdapterInput, AdapterOutput, AdapterMode, DomainType
    from day4_alignment_system import CrossDomainAligner, AlignmentConfig
except ImportError as e:
    print(f"Warning: Could not import adapters: {e}")
    print("Running in mock mode...")

class InputType(Enum):
    """输入数据类型"""
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
    MEDICAL_IMAGE = "medical_image"
    AUTO_DETECT = "auto_detect"

@dataclass
class UnifiedInput:
    """统一输入数据结构"""
    data: Any
    input_type: InputType
    domain: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class UnifiedOutput:
    """统一输出数据结构"""
    embeddings: torch.Tensor
    domain: str
    confidence: float
    logits: Optional[torch.Tensor] = None
    predictions: Optional[Any] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class BatchInferenceResult:
    """批量推理结果"""
    outputs: List[UnifiedOutput]
    total_processing_time: float
    cross_domain_similarities: Optional[Dict[str, float]] = None
    batch_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.batch_metadata is None:
            self.batch_metadata = {}

class DomainDetector:
    """自动域识别器"""

    @staticmethod
    def detect_domain(data: Any, input_type: InputType) -> str:
        """根据输入数据自动检测域"""
        if input_type == InputType.AUTO_DETECT:
            input_type = DomainDetector._detect_input_type(data)

        # 基于输入类型映射到域
        type_to_domain = {
            InputType.IMAGE: "vision",
            InputType.TEXT: "nlp",
            InputType.TABULAR: "security",
            InputType.MEDICAL_IMAGE: "medical"
        }

        return type_to_domain.get(input_type, "vision")

    @staticmethod
    def _detect_input_type(data: Any) -> InputType:
        """自动检测输入数据类型"""
        if isinstance(data, str):
            return InputType.TEXT
        elif torch.is_tensor(data):
            if len(data.shape) == 4:  # [B, C, H, W]
                if data.shape[1] == 1:  # 单通道，可能是医疗图像
                    return InputType.MEDICAL_IMAGE
                else:  # 多通道，可能是自然图像
                    return InputType.IMAGE
            elif len(data.shape) == 2:  # [B, Features]
                return InputType.TABULAR
            else:
                return InputType.IMAGE
        elif isinstance(data, (list, tuple)):
            if all(isinstance(item, str) for item in data):
                return InputType.TEXT
            else:
                return InputType.IMAGE
        else:
            return InputType.IMAGE

class UnifiedInferenceSystem:
    """统一推理系统"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adapters = {}
        self.aligner = None
        self.domain_detector = DomainDetector()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = self._setup_logger()

        # 初始化适配器
        self._initialize_adapters()

        # 初始化对齐器
        self._initialize_aligner()

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('UnifiedInference')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _initialize_adapters(self):
        """初始化所有适配器"""
        try:
            self.logger.info("正在初始化适配器...")

            # 初始化视觉适配器
            self.adapters['vision'] = VisionAdapter(
                backbone='custom',
                num_classes=10,
                output_dim=128,
                hidden_dim=256,
                device=self.device
            )

            # 初始化NLP适配器
            self.adapters['nlp'] = NLPAdapter(
                backbone='custom',
                num_classes=2,
                output_dim=128,
                device=self.device
            )

            # 初始化安全适配器
            self.adapters['security'] = SecurityAdapter(
                feature_dim=122,
                num_classes=5,
                output_dim=128,
                device=self.device
            )

            # 初始化医疗适配器
            self.adapters['medical'] = MedicalAdapter(
                modality=MedicalModalityType.XRAY,
                num_diseases=14,
                output_dim=128,
                device=self.device
            )

            # 设置为推理模式
            for adapter in self.adapters.values():
                adapter.set_mode(AdapterMode.INFERENCE)
                adapter.eval()

            self.logger.info(f"成功初始化 {len(self.adapters)} 个适配器")

        except Exception as e:
            self.logger.error(f"适配器初始化失败: {e}")
            # 创建模拟适配器
            self._create_mock_adapters()

    def _create_mock_adapters(self):
        """创建模拟适配器（用于测试）"""
        class MockAdapter(nn.Module):
            def __init__(self, domain: str):
                super().__init__()
                self.domain = domain
                self.linear = nn.Linear(128, 128)

            def __call__(self, inputs):
                if isinstance(inputs, AdapterInput):
                    data = inputs.raw_data
                else:
                    data = inputs

                # 模拟处理
                if isinstance(data, torch.Tensor):
                    batch_size = data.shape[0] if len(data.shape) > 1 else 1
                elif isinstance(data, (list, tuple)):
                    batch_size = len(data)
                else:
                    batch_size = 1

                embeddings = torch.randn(batch_size, 128, device=self.device)
                logits = torch.randn(batch_size, 10, device=self.device)
                confidence = torch.rand(batch_size, device=self.device)

                return AdapterOutput(
                    embeddings=embeddings,
                    logits=logits,
                    confidence_scores=confidence
                )

            def set_mode(self, mode):
                pass

            def eval(self):
                pass

        for domain in ['vision', 'nlp', 'security', 'medical']:
            self.adapters[domain] = MockAdapter(domain)

        self.logger.info("已创建模拟适配器")

    def _initialize_aligner(self):
        """初始化对齐器"""
        try:
            config = AlignmentConfig(
                temperature=0.07,
                margin=0.2,
                alignment_weight=1.0,
                diversity_weight=0.1
            )
            self.aligner = CrossDomainAligner(config).to(self.device)
            self.logger.info("对齐器初始化成功")
        except Exception as e:
            self.logger.warning(f"对齐器初始化失败: {e}")
            self.aligner = None

    def preprocess_input(self, data: Any, input_type: InputType = InputType.AUTO_DETECT) -> UnifiedInput:
        """预处理输入数据"""
        # 自动检测域
        domain = self.domain_detector.detect_domain(data, input_type)

        # 创建统一输入
        unified_input = UnifiedInput(
            data=data,
            input_type=input_type,
            domain=domain,
            metadata={
                'timestamp': time.time(),
                'device': str(self.device)
            }
        )

        return unified_input

    def single_inference(self, unified_input: UnifiedInput) -> UnifiedOutput:
        """单个样本推理"""
        start_time = time.time()

        try:
            # 获取对应的适配器
            adapter = self.adapters[unified_input.domain]

            # 创建适配器输入
            if unified_input.domain == 'nlp' and isinstance(unified_input.data, str):
                adapter_input = AdapterInput(raw_data=[unified_input.data])
            else:
                adapter_input = AdapterInput(raw_data=unified_input.data)

            # 推理
            with torch.no_grad():
                output = adapter(adapter_input)

            # 计算置信度（取最大值）
            confidence = output.confidence_scores.mean().item() if torch.is_tensor(output.confidence_scores) else 0.5

            # 创建统一输出
            processing_time = time.time() - start_time

            unified_output = UnifiedOutput(
                embeddings=output.embeddings,
                domain=unified_input.domain,
                confidence=confidence,
                logits=output.logits,
                predictions=None,  # 可以添加后处理得到预测结果
                processing_time=processing_time,
                metadata={
                    'input_type': unified_input.input_type,
                    'batch_size': output.embeddings.shape[0] if torch.is_tensor(output.embeddings) else 1
                }
            )

            return unified_output

        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            # 返回默认输出
            return UnifiedOutput(
                embeddings=torch.zeros(1, 128, device=self.device),
                domain=unified_input.domain,
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )

    def batch_inference(self, inputs: List[UnifiedInput]) -> BatchInferenceResult:
        """批量推理"""
        start_time = time.time()
        self.logger.info(f"开始批量推理 {len(inputs)} 个样本")

        # 按域分组
        domain_groups = {}
        for i, inp in enumerate(inputs):
            if inp.domain not in domain_groups:
                domain_groups[inp.domain] = []
            domain_groups[inp.domain].append((i, inp))

        # 并行处理每个域
        outputs = [None] * len(inputs)
        futures = []

        for domain, group_inputs in domain_groups.items():
            future = self.executor.submit(self._process_domain_batch, domain, group_inputs)
            futures.append((domain, future))

        # 收集结果
        for domain, future in futures:
            try:
                domain_outputs = future.result(timeout=30)
                for original_idx, output in domain_outputs:
                    outputs[original_idx] = output
            except Exception as e:
                self.logger.error(f"域 {domain} 批量处理失败: {e}")

        # 过滤None结果
        valid_outputs = [out for out in outputs if out is not None]

        # 计算跨域相似度（如果有多个域）
        cross_domain_similarities = None
        if len(domain_groups) > 1 and self.aligner is not None:
            cross_domain_similarities = self._compute_cross_domain_similarities(valid_outputs)

        total_time = time.time() - start_time
        self.logger.info(f"批量推理完成，耗时 {total_time:.3f}s")

        return BatchInferenceResult(
            outputs=valid_outputs,
            total_processing_time=total_time,
            cross_domain_similarities=cross_domain_similarities,
            batch_metadata={
                'total_samples': len(inputs),
                'domains_processed': list(domain_groups.keys()),
                'avg_processing_time': total_time / len(valid_outputs) if valid_outputs else 0
            }
        )

    def _process_domain_batch(self, domain: str, group_inputs: List[Tuple[int, UnifiedInput]]) -> List[Tuple[int, UnifiedOutput]]:
        """处理单个域的批量数据"""
        results = []

        for original_idx, unified_input in group_inputs:
            output = self.single_inference(unified_input)
            results.append((original_idx, output))

        return results

    def _compute_cross_domain_similarities(self, outputs: List[UnifiedOutput]) -> Dict[str, float]:
        """计算跨域相似度"""
        similarities = {}

        # 按域分组嵌入
        domain_embeddings = {}
        for output in outputs:
            if output.domain not in domain_embeddings:
                domain_embeddings[output.domain] = []
            domain_embeddings[output.domain].append(output.embeddings)

        # 计算域间相似度
        domains = list(domain_embeddings.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1, domain2 = domains[i], domains[j]

                # 取第一个样本计算相似度
                if domain_embeddings[domain1] and domain_embeddings[domain2]:
                    emb1 = domain_embeddings[domain1][0][0] if len(domain_embeddings[domain1][0].shape) > 1 else domain_embeddings[domain1][0]
                    emb2 = domain_embeddings[domain2][0][0] if len(domain_embeddings[domain2][0].shape) > 1 else domain_embeddings[domain2][0]

                    similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
                    similarities[f"{domain1}-{domain2}"] = similarity

        return similarities

    async def async_inference(self, unified_input: UnifiedInput) -> UnifiedOutput:
        """异步推理"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.single_inference, unified_input)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'device': str(self.device),
            'adapters_loaded': list(self.adapters.keys()),
            'aligner_available': self.aligner is not None,
            'memory_usage': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'executor_status': {
                'max_workers': self.executor._max_workers,
                'threads_count': len(self.executor._threads)
            }
        }

    def shutdown(self):
        """关闭系统"""
        self.executor.shutdown(wait=True)
        self.logger.info("统一推理系统已关闭")

# 使用示例和测试
class UnifiedInferenceDemo:
    """统一推理系统演示"""

    def __init__(self):
        self.system = UnifiedInferenceSystem()

    def demo_single_inference(self):
        """单样本推理演示"""
        print("\n=== 单样本推理演示 ===")

        # 测试不同类型的输入
        test_cases = [
            ("This is a test sentence", InputType.TEXT),
            (torch.randn(1, 3, 224, 224), InputType.IMAGE),
            (torch.randn(1, 122), InputType.TABULAR),
            (torch.randn(1, 1, 224, 224), InputType.MEDICAL_IMAGE)
        ]

        for i, (data, input_type) in enumerate(test_cases):
            print(f"\n测试案例 {i+1}: {input_type.value}")

            # 预处理输入
            unified_input = self.system.preprocess_input(data, input_type)
            print(f"  检测到域: {unified_input.domain}")

            # 推理
            output = self.system.single_inference(unified_input)
            print(f"  输出形状: {output.embeddings.shape}")
            print(f"  置信度: {output.confidence:.3f}")
            print(f"  处理时间: {output.processing_time:.4f}s")

    def demo_batch_inference(self):
        """批量推理演示"""
        print("\n=== 批量推理演示 ===")

        # 创建混合批量数据
        batch_inputs = []

        # 添加不同类型的数据
        batch_inputs.extend([
            self.system.preprocess_input("Sample text 1", InputType.TEXT),
            self.system.preprocess_input("Sample text 2", InputType.TEXT),
            self.system.preprocess_input(torch.randn(2, 3, 224, 224), InputType.IMAGE),
            self.system.preprocess_input(torch.randn(3, 122), InputType.TABULAR),
            self.system.preprocess_input(torch.randn(1, 1, 224, 224), InputType.MEDICAL_IMAGE)
        ])

        print(f"批量大小: {len(batch_inputs)}")
        print(f"涉及域: {set(inp.domain for inp in batch_inputs)}")

        # 批量推理
        result = self.system.batch_inference(batch_inputs)

        print(f"\n批量推理结果:")
        print(f"  总处理时间: {result.total_processing_time:.3f}s")
        print(f"  平均处理时间: {result.batch_metadata['avg_processing_time']:.4f}s")
        print(f"  成功处理样本数: {len(result.outputs)}")

        if result.cross_domain_similarities:
            print(f"  跨域相似度:")
            for pair, similarity in result.cross_domain_similarities.items():
                print(f"    {pair}: {similarity:.3f}")

    def demo_async_inference(self):
        """异步推理演示"""
        print("\n=== 异步推理演示 ===")

        async def run_async_demo():
            # 创建多个异步任务
            tasks = []
            test_inputs = [
                self.system.preprocess_input(f"Async text {i}", InputType.TEXT)
                for i in range(5)
            ]

            for inp in test_inputs:
                task = self.system.async_inference(inp)
                tasks.append(task)

            # 等待所有任务完成
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            print(f"异步处理 {len(results)} 个样本")
            print(f"总时间: {total_time:.3f}s")
            print(f"平均时间: {total_time/len(results):.4f}s")

            for i, result in enumerate(results):
                print(f"  样本 {i+1}: 置信度={result.confidence:.3f}")

        # 运行异步演示
        asyncio.run(run_async_demo())

    def demo_system_status(self):
        """系统状态演示"""
        print("\n=== 系统状态 ===")
        status = self.system.get_system_status()

        for key, value in status.items():
            print(f"  {key}: {value}")

    def run_all_demos(self):
        """运行所有演示"""
        print("UniMatch-Clip 统一推理系统演示")
        print("=" * 50)

        self.demo_single_inference()
        self.demo_batch_inference()
        self.demo_async_inference()
        self.demo_system_status()

        print("\n演示完成!")
        self.system.shutdown()

if __name__ == "__main__":
    # 运行演示
    demo = UnifiedInferenceDemo()
    demo.run_all_demos()