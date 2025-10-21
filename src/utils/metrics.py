"""
跨域置信度计算器 - UniMatch-Clip项目核心组件

统一四个领域的置信度计算接口：
- Vision: 目标检测/分类置信度 (max_prob + bbox_iou + spatial_consistency)
- NLP: 文本分类置信度 (softmax_entropy + attention_consistency + linguistic_features)
- Security: 异常检测置信度 (anomaly_score + ensemble_agreement + feature_stability)
- Medical: 诊断置信度 (diagnosis_prob + uncertainty_estimation + clinical_consistency)

目标：将不同领域的模型输出转换为 [0,1] 统一置信度分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings


@dataclass
class ConfidenceOutput:
    """置信度计算结果"""
    confidence: float  # 主置信度分数 [0,1]
    components: Dict[str, float]  # 各组件分数
    metadata: Dict[str, Any]  # 额外信息
    domain: str  # 所属领域


class BaseConfidenceCalculator(ABC):
    """置信度计算器基类"""

    def __init__(self, domain: str, normalize: bool = True):
        self.domain = domain
        self.normalize = normalize
        self.logger = logging.getLogger(f"Confidence_{domain}")

    @abstractmethod
    def compute_confidence(self, model_output: Dict[str, torch.Tensor],
                           inputs: Optional[torch.Tensor] = None,
                           **kwargs) -> ConfidenceOutput:
        """计算置信度的抽象方法"""
        pass

    def _normalize_score(self, score: float, min_val: float = 0.0,
                         max_val: float = 1.0) -> float:
        """归一化分数到[0,1]"""
        if not self.normalize:
            return score
        return np.clip((score - min_val) / (max_val - min_val), 0.0, 1.0)


class VisionConfidenceCalculator(BaseConfidenceCalculator):
    """
    Vision领域置信度计算器

    支持场景：
    - 图像分类: softmax概率 + 特征一致性
    - 目标检测: bbox置信度 + IoU + 空间一致性
    - 语义分割: 像素级置信度 + 区域一致性
    """

    def __init__(self, task_type='classification', iou_threshold=0.5):
        super().__init__('vision')
        self.task_type = task_type  # 'classification', 'detection', 'segmentation'
        self.iou_threshold = iou_threshold

    def compute_confidence(self, model_output: Dict[str, torch.Tensor],
                           inputs: Optional[torch.Tensor] = None,
                           **kwargs) -> ConfidenceOutput:
        """
        计算Vision置信度

        Args:
            model_output: 模型输出字典，包含：
                - 'logits': [B, num_classes] 分类logits
                - 'boxes': [B, N, 4] bbox坐标 (可选)
                - 'scores': [B, N] bbox置信度分数 (可选)
                - 'features': [B, D] 特征向量 (可选)
            inputs: [B, C, H, W] 输入图像张量 (可选)

        Returns:
            ConfidenceOutput对象
        """

        components = {}
        metadata = {}

        if self.task_type == 'classification':
            confidence = self._compute_classification_confidence(
                model_output, inputs, components, metadata
            )
        elif self.task_type == 'detection':
            confidence = self._compute_detection_confidence(
                model_output, inputs, components, metadata
            )
        else:
            # 默认分类方式
            confidence = self._compute_classification_confidence(
                model_output, inputs, components, metadata
            )

        return ConfidenceOutput(
            confidence=confidence,
            components=components,
            metadata=metadata,
            domain='vision'
        )

    def _compute_classification_confidence(self, model_output: Dict,
                                           inputs: Optional[torch.Tensor],
                                           components: Dict, metadata: Dict) -> float:
        """计算分类任务置信度"""

        logits = model_output.get('logits')
        if logits is None:
            raise ValueError("分类任务需要'logits'输出")

        # 1. 最大概率分数 (主要指标)
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]  # [B]
        max_prob_score = max_probs.mean().item()
        components['max_probability'] = max_prob_score

        # 2. 熵分数 (不确定性度量)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B]
        max_entropy = np.log(probs.size(-1))  # log(num_classes)
        entropy_score = 1.0 - (entropy.mean().item() / max_entropy)
        components['entropy_confidence'] = entropy_score

        # 3. 预测边际 (top1与top2差距)
        sorted_probs = torch.sort(probs, dim=-1, descending=True)[0]
        if sorted_probs.size(-1) > 1:
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # [B]
            margin_score = margin.mean().item()
        else:
            margin_score = max_prob_score
        components['prediction_margin'] = margin_score

        # 4. 特征一致性 (如果有特征)
        feature_consistency = 0.0
        if 'features' in model_output:
            features = model_output['features']  # [B, D]
            if features.size(0) > 1:
                # 计算批次内特征相似度
                normalized_features = F.normalize(features, dim=-1)
                similarity_matrix = torch.mm(normalized_features, normalized_features.t())
                # 排除对角线，计算平均相似度
                mask = torch.eye(features.size(0), device=features.device).bool()
                similarity_matrix = similarity_matrix.masked_fill(mask, 0.0)
                feature_consistency = similarity_matrix.mean().item()
        components['feature_consistency'] = feature_consistency

        # 5. 综合置信度计算 (加权平均)
        weights = {
            'max_probability': 0.4,
            'entropy_confidence': 0.3,
            'prediction_margin': 0.2,
            'feature_consistency': 0.1
        }

        final_confidence = sum(
            components[key] * weights[key] for key in weights.keys()
        )

        # 记录元数据
        metadata.update({
            'task_type': 'classification',
            'num_classes': probs.size(-1),
            'batch_size': probs.size(0),
            'weights_used': weights
        })

        return final_confidence

    def _compute_detection_confidence(self, model_output: Dict,
                                      inputs: Optional[torch.Tensor],
                                      components: Dict, metadata: Dict) -> float:
        """计算目标检测置信度"""

        boxes = model_output.get('boxes')  # [B, N, 4]
        scores = model_output.get('scores')  # [B, N]

        if boxes is None or scores is None:
            raise ValueError("检测任务需要'boxes'和'scores'输出")

        # 1. 检测分数 (主要指标)
        detection_scores = scores.mean().item()
        components['detection_scores'] = detection_scores

        # 2. bbox质量评估 (基于面积和位置)
        box_areas = self._compute_box_areas(boxes)  # [B, N]
        # 过滤掉面积过小的bbox
        valid_boxes = box_areas > 0.01  # 至少占图像1%
        if valid_boxes.sum() > 0:
            box_quality = (box_areas * valid_boxes.float()).sum() / valid_boxes.sum()
            box_quality = box_quality.item()
        else:
            box_quality = 0.0
        components['box_quality'] = box_quality

        # 3. 空间一致性 (检测框分布)
        spatial_consistency = self._compute_spatial_consistency(boxes)
        components['spatial_consistency'] = spatial_consistency

        # 4. NMS后的保留率 (间接质量指标)
        nms_retention = self._estimate_nms_retention(boxes, scores)
        components['nms_retention'] = nms_retention

        # 5. 综合置信度
        weights = {
            'detection_scores': 0.5,
            'box_quality': 0.2,
            'spatial_consistency': 0.15,
            'nms_retention': 0.15
        }

        final_confidence = sum(
            components[key] * weights[key] for key in weights.keys()
        )

        metadata.update({
            'task_type': 'detection',
            'num_detections': boxes.size(1),
            'batch_size': boxes.size(0),
            'iou_threshold': self.iou_threshold
        })

        return final_confidence

    def _compute_box_areas(self, boxes: torch.Tensor) -> torch.Tensor:
        """计算bbox面积 [B, N]"""
        # boxes: [B, N, 4] (x1, y1, x2, y2)
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        areas = widths * heights
        return torch.clamp(areas, min=0.0)

    def _compute_spatial_consistency(self, boxes: torch.Tensor) -> float:
        """计算检测框空间分布一致性"""
        # 简化实现：计算bbox中心点的分布均匀性
        centers = boxes[:, :, :2] + (boxes[:, :, 2:] - boxes[:, :, :2]) / 2
        # 计算中心点的标准差，作为分布一致性的逆指标
        std_x = centers[:, :, 0].std().item()
        std_y = centers[:, :, 1].std().item()
        consistency = 1.0 / (1.0 + std_x + std_y)
        return consistency

    def _estimate_nms_retention(self, boxes: torch.Tensor,
                                scores: torch.Tensor) -> float:
        """估算NMS后的保留率"""
        # 简化实现：高分检测框的占比
        high_score_mask = scores > 0.5
        retention_rate = high_score_mask.float().mean().item()
        return retention_rate


class NLPConfidenceCalculator(BaseConfidenceCalculator):
    """
    NLP领域置信度计算器

    支持场景：
    - 文本分类: softmax置信度 + attention一致性
    - 序列标注: token级置信度聚合
    - 生成任务: 生成质量置信度
    """

    def __init__(self, task_type='classification', attention_layers=None):
        super().__init__('nlp')
        self.task_type = task_type
        self.attention_layers = attention_layers or [-1]  # 使用最后一层attention

    def compute_confidence(self, model_output: Dict[str, torch.Tensor],
                           inputs: Optional[torch.Tensor] = None,
                           **kwargs) -> ConfidenceOutput:
        """
        计算NLP置信度

        Args:
            model_output: 模型输出字典，包含：
                - 'logits': [B, seq_len, vocab_size] 或 [B, num_classes]
                - 'attentions': List of [B, num_heads, seq_len, seq_len] (可选)
                - 'hidden_states': [B, seq_len, hidden_size] (可选)
            inputs: [B, seq_len] token ids (可选)

        Returns:
            ConfidenceOutput对象
        """

        components = {}
        metadata = {}

        if self.task_type == 'classification':
            confidence = self._compute_text_classification_confidence(
                model_output, inputs, components, metadata
            )
        elif self.task_type == 'generation':
            confidence = self._compute_generation_confidence(
                model_output, inputs, components, metadata
            )
        else:
            confidence = self._compute_text_classification_confidence(
                model_output, inputs, components, metadata
            )

        return ConfidenceOutput(
            confidence=confidence,
            components=components,
            metadata=metadata,
            domain='nlp'
        )

    def _compute_text_classification_confidence(self, model_output: Dict,
                                                inputs: Optional[torch.Tensor],
                                                components: Dict, metadata: Dict) -> float:
        """计算文本分类置信度"""

        logits = model_output.get('logits')
        if logits is None:
            raise ValueError("文本分类需要'logits'输出")

        # 如果是序列logits，取平均或最后一个token
        if logits.dim() == 3:  # [B, seq_len, num_classes]
            logits = logits[:, -1, :]  # 取最后一个token

        # 1. 基础分类置信度 (与Vision类似)
        probs = F.softmax(logits, dim=-1)
        max_prob_score = probs.max(dim=-1)[0].mean().item()
        components['max_probability'] = max_prob_score

        # 2. 语言学特征一致性
        linguistic_consistency = 0.0
        if 'hidden_states' in model_output:
            hidden_states = model_output['hidden_states']  # [B, seq_len, D]
            linguistic_consistency = self._compute_linguistic_consistency(hidden_states)
        components['linguistic_consistency'] = linguistic_consistency

        # 3. 注意力一致性
        attention_consistency = 0.0
        if 'attentions' in model_output:
            attentions = model_output['attentions']
            attention_consistency = self._compute_attention_consistency(attentions)
        components['attention_consistency'] = attention_consistency

        # 4. 词汇多样性分数 (如果有输入token)
        vocabulary_confidence = 0.0
        if inputs is not None:
            vocabulary_confidence = self._compute_vocabulary_confidence(inputs)
        components['vocabulary_confidence'] = vocabulary_confidence

        # 5. 综合置信度
        weights = {
            'max_probability': 0.4,
            'linguistic_consistency': 0.25,
            'attention_consistency': 0.25,
            'vocabulary_confidence': 0.1
        }

        final_confidence = sum(
            components[key] * weights[key] for key in weights.keys()
        )

        metadata.update({
            'task_type': 'text_classification',
            'num_classes': probs.size(-1),
            'sequence_length': hidden_states.size(1) if 'hidden_states' in model_output else 0
        })

        return final_confidence

    def _compute_linguistic_consistency(self, hidden_states: torch.Tensor) -> float:
        """计算语言学特征一致性"""
        # [B, seq_len, hidden_size]
        # 计算序列内hidden states的余弦相似度
        normalized_states = F.normalize(hidden_states, dim=-1)
        # 计算相邻token的相似度
        similarities = []
        for i in range(hidden_states.size(1) - 1):
            sim = F.cosine_similarity(
                normalized_states[:, i, :],
                normalized_states[:, i + 1, :],
                dim=-1
            )
            similarities.append(sim.mean().item())

        if similarities:
            consistency = np.mean(similarities)
        else:
            consistency = 0.0

        return consistency

    def _compute_attention_consistency(self, attentions: List[torch.Tensor]) -> float:
        """计算注意力模式一致性"""
        if not attentions:
            return 0.0

        # 取指定层的attention
        layer_idx = self.attention_layers[0]
        if layer_idx >= len(attentions):
            layer_idx = -1

        attention = attentions[layer_idx]  # [B, num_heads, seq_len, seq_len]

        # 计算不同head之间的一致性
        B, num_heads, seq_len, _ = attention.shape
        head_similarities = []

        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                # 计算head i和head j的注意力模式相似度
                head_i = attention[:, i, :, :].flatten(1)  # [B, seq_len^2]
                head_j = attention[:, j, :, :].flatten(1)  # [B, seq_len^2]

                sim = F.cosine_similarity(head_i, head_j, dim=1).mean().item()
                head_similarities.append(sim)

        if head_similarities:
            consistency = np.mean(head_similarities)
        else:
            consistency = 0.0

        return consistency

    def _compute_vocabulary_confidence(self, input_ids: torch.Tensor) -> float:
        """计算词汇多样性置信度"""
        # 简单实现：计算unique token的比例
        batch_size, seq_len = input_ids.shape
        unique_ratios = []

        for i in range(batch_size):
            tokens = input_ids[i]
            unique_tokens = torch.unique(tokens)
            unique_ratio = len(unique_tokens) / seq_len
            unique_ratios.append(unique_ratio)

        return np.mean(unique_ratios)

    def _compute_generation_confidence(self, model_output: Dict,
                                       inputs: Optional[torch.Tensor],
                                       components: Dict, metadata: Dict) -> float:
        """计算生成任务置信度"""
        # 简化实现，主要基于logits
        logits = model_output.get('logits')
        if logits is None:
            return 0.0

        # 对于生成任务，计算序列级别的置信度
        probs = F.softmax(logits, dim=-1)  # [B, seq_len, vocab_size]

        # 每个position的最大概率
        max_probs = probs.max(dim=-1)[0]  # [B, seq_len]

        # 序列级置信度：所有position概率的几何平均
        sequence_confidence = torch.exp(torch.log(max_probs + 1e-8).mean(dim=-1))
        final_confidence = sequence_confidence.mean().item()

        components['sequence_confidence'] = final_confidence
        metadata['task_type'] = 'generation'

        return final_confidence


class SecurityConfidenceCalculator(BaseConfidenceCalculator):
    """
    Security领域置信度计算器

    支持场景：
    - 异常检测: 异常分数置信度
    - 入侵检测: 威胁级别置信度
    - 恶意软件检测: 检测置信度
    """

    def __init__(self, threshold=0.5):
        super().__init__('security')
        self.threshold = threshold

    def compute_confidence(self, model_output: Dict[str, torch.Tensor],
                           inputs: Optional[torch.Tensor] = None,
                           **kwargs) -> ConfidenceOutput:
        """计算Security置信度"""

        components = {}
        metadata = {}

        # 异常检测置信度
        anomaly_scores = model_output.get('anomaly_scores')  # [B]
        if anomaly_scores is not None:
            # 异常分数转换为置信度
            confidence_scores = torch.sigmoid(anomaly_scores)
            components['anomaly_confidence'] = confidence_scores.mean().item()

        # 威胁级别置信度
        threat_logits = model_output.get('threat_logits')  # [B, num_threat_types]
        if threat_logits is not None:
            threat_probs = F.softmax(threat_logits, dim=-1)
            threat_confidence = threat_probs.max(dim=-1)[0].mean().item()
            components['threat_confidence'] = threat_confidence

        # 综合置信度
        final_confidence = np.mean(list(components.values())) if components else 0.0

        metadata.update({
            'domain': 'security',
            'threshold': self.threshold,
            'components_count': len(components)
        })

        return ConfidenceOutput(
            confidence=final_confidence,
            components=components,
            metadata=metadata,
            domain='security'
        )


class MedicalConfidenceCalculator(BaseConfidenceCalculator):
    """
    Medical领域置信度计算器

    支持场景：
    - 医学图像诊断: 疾病分类置信度
    - 症状分析: 诊断置信度
    - 药物推荐: 推荐置信度
    """

    def __init__(self, uncertainty_method='mc_dropout'):
        super().__init__('medical')
        self.uncertainty_method = uncertainty_method

    def compute_confidence(self, model_output: Dict[str, torch.Tensor],
                           inputs: Optional[torch.Tensor] = None,
                           **kwargs) -> ConfidenceOutput:
        """计算Medical置信度"""

        components = {}
        metadata = {}

        # 诊断置信度
        diagnosis_logits = model_output.get('diagnosis_logits')  # [B, num_diseases]
        if diagnosis_logits is not None:
            diagnosis_probs = F.softmax(diagnosis_logits, dim=-1)
            diagnosis_confidence = diagnosis_probs.max(dim=-1)[0].mean().item()
            components['diagnosis_confidence'] = diagnosis_confidence

        # 不确定性估计
        uncertainty_scores = model_output.get('uncertainty')  # [B]
        if uncertainty_scores is not None:
            # 不确定性越小，置信度越高
            uncertainty_confidence = 1.0 - uncertainty_scores.mean().item()
            components['uncertainty_confidence'] = uncertainty_confidence

        # 临床一致性 (如果有多个预测)
        if 'ensemble_predictions' in model_output:
            clinical_consistency = self._compute_clinical_consistency(
                model_output['ensemble_predictions']
            )
            components['clinical_consistency'] = clinical_consistency

        # 综合置信度
        final_confidence = np.mean(list(components.values())) if components else 0.0

        metadata.update({
            'domain': 'medical',
            'uncertainty_method': self.uncertainty_method,
            'components_count': len(components)
        })

        return ConfidenceOutput(
            confidence=final_confidence,
            components=components,
            metadata=metadata,
            domain='medical'
        )

    def _compute_clinical_consistency(self, ensemble_predictions: torch.Tensor) -> float:
        """计算临床一致性"""
        # [num_models, B, num_classes]
        # 计算不同模型预测的一致性
        predicted_classes = ensemble_predictions.argmax(dim=-1)  # [num_models, B]

        # 计算每个样本的预测一致性
        consistency_scores = []
        for i in range(predicted_classes.size(1)):  # 遍历batch
            sample_predictions = predicted_classes[:, i]  # [num_models]
            # 计算众数出现的频率
            unique_preds, counts = torch.unique(sample_predictions, return_counts=True)
            max_count = counts.max().item()
            consistency = max_count / predicted_classes.size(0)
            consistency_scores.append(consistency)

        return np.mean(consistency_scores)


class UniversalConfidenceCalculator:
    """
    通用置信度计算器 - UniMatch-Clip的核心组件

    统一管理四个领域的置信度计算，提供统一接口
    """

    def __init__(self):
        # 初始化各领域计算器
        self.calculators = {
            'vision': VisionConfidenceCalculator(),
            'nlp': NLPConfidenceCalculator(),
            'security': SecurityConfidenceCalculator(),
            'medical': MedicalConfidenceCalculator()
        }

        self.logger = logging.getLogger("UniversalConfidence")

    def compute_confidence(self, domain: str, model_output: Dict[str, torch.Tensor],
                           inputs: Optional[torch.Tensor] = None,
                           **kwargs) -> ConfidenceOutput:
        """
        统一置信度计算接口

        Args:
            domain: 领域名称 ('vision', 'nlp', 'security', 'medical')
            model_output: 模型输出字典
            inputs: 输入张量 (可选)
            **kwargs: 其他参数

        Returns:
            ConfidenceOutput对象
        """

        if domain not in self.calculators:
            raise ValueError(f"不支持的领域: {domain}")

        calculator = self.calculators[domain]

        try:
            result = calculator.compute_confidence(model_output, inputs, **kwargs)
            self.logger.debug(f"Domain {domain} confidence: {result.confidence:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"计算{domain}置信度失败: {e}")
            # 返回默认低置信度
            return ConfidenceOutput(
                confidence=0.1,
                components={'error': 0.1},
                metadata={'error': str(e)},
                domain=domain
            )

    def batch_compute_confidence(self, domain: str,
                                 model_outputs: List[Dict[str, torch.Tensor]],
                                 inputs: Optional[List[torch.Tensor]] = None) -> List[ConfidenceOutput]:
        """批量计算置信度"""

        results = []
        inputs_list = inputs or [None] * len(model_outputs)

        for i, (output, input_tensor) in enumerate(zip(model_outputs, inputs_list)):
            try:
                result = self.compute_confidence(domain, output, input_tensor)
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量计算第{i}个样本失败: {e}")
                results.append(ConfidenceOutput(
                    confidence=0.1,
                    components={'error': 0.1},
                    metadata={'error': str(e), 'batch_index': i},
                    domain=domain
                ))

        return results

    def get_supported_domains(self) -> List[str]:
        """获取支持的领域列表"""
        return list(self.calculators.keys())

    def update_domain_calculator(self, domain: str, calculator: BaseConfidenceCalculator):
        """更新特定领域的计算器"""
        self.calculators[domain] = calculator
        self.logger.info(f"更新了{domain}领域的置信度计算器")


# =====================================================
# 测试和使用示例
# =====================================================

def test_confidence_calculators():
    """测试各领域置信度计算器"""

    print("🧪 测试跨域置信度计算器...")

    # 创建通用计算器
    universal_calc = UniversalConfidenceCalculator()

    # 1. 测试Vision领域
    print("\n📊 测试Vision置信度...")
    vision_output = {
        'logits': torch.randn(4, 10),  # 4个样本，10类分类
        'features': torch.randn(4, 256)  # 特征向量
    }

    vision_result = universal_calc.compute_confidence('vision', vision_output)
    print(f"Vision置信度: {vision_result.confidence:.4f}")
    print(f"Vision组件: {vision_result.components}")

    # 2. 测试NLP领域
    print("\n📝 测试NLP置信度...")
    nlp_output = {
        'logits': torch.randn(2, 5),  # 2个样本，5类分类
        'hidden_states': torch.randn(2, 20, 768),  # [B, seq_len, hidden_size]
        'attentions': [torch.randn(2, 12, 20, 20)]  # 单层attention
    }

    nlp_inputs = torch.randint(0, 1000, (2, 20))  # token ids
    nlp_result = universal_calc.compute_confidence('nlp', nlp_output, nlp_inputs)
    print(f"NLP置信度: {nlp_result.confidence:.4f}")
    print(f"NLP组件: {nlp_result.components}")

    # 3. 测试Security领域
    print("\n🔒 测试Security置信度...")
    security_output = {
        'anomaly_scores': torch.randn(3),  # 3个样本的异常分数
        'threat_logits': torch.randn(3, 4)  # 4种威胁类型
    }

    security_result = universal_calc.compute_confidence('security', security_output)
    print(f"Security置信度: {security_result.confidence:.4f}")
    print(f"Security组件: {security_result.components}")

    # 4. 测试Medical领域
    print("\n🏥 测试Medical置信度...")
    medical_output = {
        'diagnosis_logits': torch.randn(2, 6),  # 6种疾病
        'uncertainty': torch.rand(2) * 0.3,  # 不确定性分数 [0, 0.3]
    }

    medical_result = universal_calc.compute_confidence('medical', medical_output)
    print(f"Medical置信度: {medical_result.confidence:.4f}")
    print(f"Medical组件: {medical_result.components}")

    # 5. 测试批量计算
    print("\n📦 测试批量计算...")
    batch_outputs = [vision_output, vision_output]  # 2个vision样本
    batch_results = universal_calc.batch_compute_confidence('vision', batch_outputs)
    print(f"批量结果: {[r.confidence for r in batch_results]}")

    print("\n✅ 置信度计算器测试完成!")

    return True


def demonstrate_confidence_usage():
    """演示如何在实际场景中使用置信度计算器"""

    print("\n🎯 置信度计算器实际使用演示...")

    calc = UniversalConfidenceCalculator()

    # 模拟实际使用场景
    scenarios = [
        {
            'domain': 'vision',
            'description': '图像分类 - 猫狗识别',
            'output': {
                'logits': torch.tensor([[2.1, -0.5]]),  # 高置信度预测猫
                'features': torch.randn(1, 512)
            }
        },
        {
            'domain': 'nlp',
            'description': '情感分析 - 积极/消极',
            'output': {
                'logits': torch.tensor([[0.1, 2.8]]),  # 高置信度积极
                'hidden_states': torch.randn(1, 15, 768)
            }
        },
        {
            'domain': 'security',
            'description': '异常检测 - 网络流量',
            'output': {
                'anomaly_scores': torch.tensor([3.2]),  # 高异常分数
            }
        }
    ]

    for scenario in scenarios:
        result = calc.compute_confidence(
            scenario['domain'],
            scenario['output']
        )

        print(f"\n场景: {scenario['description']}")
        print(f"领域: {scenario['domain']}")
        print(f"置信度: {result.confidence:.4f}")

        # 根据置信度给出建议
        if result.confidence > 0.8:
            print("✅ 高置信度预测，可直接使用")
        elif result.confidence > 0.5:
            print("⚠️  中等置信度，建议人工审核")
        else:
            print("❌ 低置信度，标记为困难样本")

    print("\n🎉 使用演示完成!")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    print("🚀 跨域置信度计算器测试启动")
    print("=" * 50)

    # 运行测试
    test_success = test_confidence_calculators()

    if test_success:
        print("\n🎯 运行使用演示...")
        demonstrate_confidence_usage()

        print("\n🎉 置信度计算器完全就绪!")
        print("✅ 可以进入困难样本选择器开发阶段")
    else:
        print("\n❌ 需要调试置信度计算器")