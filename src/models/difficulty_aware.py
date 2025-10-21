#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniMatch-Clip 核心算法重构：基于不确定性的困难样本检测器
使用蒙特卡洛dropout和贝叶斯深度学习理论进行困难样本检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from collections import defaultdict

@dataclass
class UncertaintyConfig:
    """不确定性检测配置"""
    num_mc_samples: int = 20  # 蒙特卡洛采样次数
    dropout_rate: float = 0.1  # Dropout概率
    uncertainty_threshold: float = 0.7  # 困难样本阈值
    entropy_weight: float = 0.5  # 熵权重
    variance_weight: float = 0.5  # 方差权重
    top_k_hard_samples: int = 32  # 选择的困难样本数量
    adaptive_threshold: bool = True  # 自适应阈值调节
    calibration_samples: int = 1000  # 校准样本数量

class BayesianModule(nn.Module):
    """贝叶斯神经网络模块"""

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

        # 标准神经网络层
        self.linear = nn.Linear(input_dim, output_dim)

        # 贝叶斯权重参数化
        self.weight_mu = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.weight_sigma = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

        self.bias_mu = nn.Parameter(torch.randn(output_dim) * 0.1)
        self.bias_sigma = nn.Parameter(torch.randn(output_dim) * 0.1)

        # 先验分布参数
        self.prior_weight_mu = 0.0
        self.prior_weight_sigma = 1.0
        self.prior_bias_mu = 0.0
        self.prior_bias_sigma = 1.0

    def sample_weights(self):
        """采样权重"""
        weight_eps = torch.randn_like(self.weight_sigma)
        bias_eps = torch.randn_like(self.bias_sigma)

        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_sigma)) * weight_eps
        bias = self.bias_mu + torch.log1p(torch.exp(self.bias_sigma)) * bias_eps

        return weight, bias

    def kl_divergence(self):
        """计算KL散度"""
        # 权重的KL散度
        weight_var = torch.log1p(torch.exp(self.weight_sigma)) ** 2
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu - self.prior_weight_mu) ** 2 / self.prior_weight_sigma ** 2 +
            weight_var / self.prior_weight_sigma ** 2 -
            torch.log(weight_var) + torch.log(torch.tensor(self.prior_weight_sigma ** 2)) - 1
        )

        # 偏置的KL散度
        bias_var = torch.log1p(torch.exp(self.bias_sigma)) ** 2
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu - self.prior_bias_mu) ** 2 / self.prior_bias_sigma ** 2 +
            bias_var / self.prior_bias_sigma ** 2 -
            torch.log(bias_var) + torch.log(torch.tensor(self.prior_bias_sigma ** 2)) - 1
        )

        return weight_kl + bias_kl

    def forward(self, x, sample_weights=False):
        if sample_weights:
            weight, bias = self.sample_weights()
            return F.linear(x, weight, bias)
        else:
            return self.linear(x)

class MonteCarloDropout(nn.Module):
    """蒙特卡洛Dropout层"""

    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        return F.dropout(x, p=self.dropout_rate, training=True)  # 始终启用dropout

class UncertaintyEstimator(nn.Module):
    """不确定性估计网络"""

    def __init__(self, input_dim: int, config: UncertaintyConfig):
        super().__init__()
        self.config = config

        # 不确定性估计网络
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            MonteCarloDropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            MonteCarloDropout(config.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            MonteCarloDropout(config.dropout_rate),
            nn.Linear(64, 2)  # 输出均值和方差
        )

        # 贝叶斯分类头
        self.bayesian_classifier = BayesianModule(input_dim, 64, config.dropout_rate)
        self.final_classifier = nn.Linear(64, 1)

    def forward(self, x, return_uncertainty=True):
        if return_uncertainty:
            return self.forward_with_uncertainty(x)
        else:
            return self.uncertainty_net(x)

    def forward_with_uncertainty(self, x):
        """前向传播并返回不确定性"""
        # 标准前向传播
        logits = self.uncertainty_net(x)

        if not return_uncertainty:
            return logits

        # 蒙特卡洛采样估计不确定性
        mc_samples = []
        for _ in range(self.config.num_mc_samples):
            sample_logits = self.uncertainty_net(x)
            mc_samples.append(sample_logits)

        mc_samples = torch.stack(mc_samples, dim=0)  # (num_samples, batch_size, output_dim)

        # 计算预测均值和方差
        prediction_mean = torch.mean(mc_samples, dim=0)
        prediction_var = torch.var(mc_samples, dim=0)

        return {
            'logits': logits,
            'prediction_mean': prediction_mean,
            'prediction_variance': prediction_var,
            'mc_samples': mc_samples
        }

class UncertaintyHardSampleDetector(nn.Module):
    """基于不确定性的困难样本检测器"""

    def __init__(self, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # 域特定的不确定性估计器
        self.domain_estimators = nn.ModuleDict({
            'vision': UncertaintyEstimator(256, config),    # 假设对齐后的特征维度为256
            'nlp': UncertaintyEstimator(256, config),
            'security': UncertaintyEstimator(256, config),
            'medical': UncertaintyEstimator(256, config)
        })

        # 跨域不确定性聚合器
        self.cross_domain_aggregator = nn.Sequential(
            nn.Linear(256 * 4, 512),  # 4个域的特征
            nn.ReLU(),
            MonteCarloDropout(config.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            MonteCarloDropout(config.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 自适应阈值网络
        if config.adaptive_threshold:
            self.threshold_net = nn.Sequential(
                nn.Linear(8, 32),  # 输入各种统计量
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        # 困难度历史记录 (用于校准)
        self.difficulty_history = defaultdict(list)
        self.calibrated_thresholds = {}

    def compute_epistemic_uncertainty(self, predictions: torch.Tensor) -> torch.Tensor:
        """计算认识不确定性 (模型不确定性)"""
        # 使用预测方差作为认识不确定性的度量
        return torch.var(predictions, dim=0)

    def compute_aleatoric_uncertainty(self, model_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算偶然不确定性 (数据不确定性)"""
        # 从模型输出中提取预测方差
        if 'prediction_variance' in model_output:
            return model_output['prediction_variance']
        else:
            # 如果没有显式方差，使用logits的不确定性
            logits = model_output.get('logits', model_output.get('prediction_mean'))
            softmax_probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-8), dim=-1)
            return entropy

    def compute_predictive_entropy(self, mc_samples: torch.Tensor) -> torch.Tensor:
        """计算预测熵"""
        # 对蒙特卡洛样本计算平均预测分布
        mean_probs = torch.mean(F.softmax(mc_samples, dim=-1), dim=0)

        # 计算熵
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        return entropy

    def compute_mutual_information(self, mc_samples: torch.Tensor) -> torch.Tensor:
        """计算互信息 (认识不确定性的另一种度量)"""
        # 预测熵
        predictive_entropy = self.compute_predictive_entropy(mc_samples)

        # 期望熵
        sample_entropies = []
        for i in range(mc_samples.size(0)):
            probs = F.softmax(mc_samples[i], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            sample_entropies.append(entropy)

        expected_entropy = torch.mean(torch.stack(sample_entropies), dim=0)

        # 互信息 = 预测熵 - 期望熵
        mutual_information = predictive_entropy - expected_entropy

        return mutual_information

    def compute_confidence_interval(self, mc_samples: torch.Tensor, confidence: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算预测置信区间"""
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = torch.quantile(mc_samples, lower_percentile / 100, dim=0)
        upper_bound = torch.quantile(mc_samples, upper_percentile / 100, dim=0)

        return lower_bound, upper_bound

    def detect_hard_samples_single_domain(self, embeddings: torch.Tensor,
                                        domain: str,
                                        labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """单域困难样本检测"""
        if domain not in self.domain_estimators:
            raise ValueError(f"Unknown domain: {domain}")

        estimator = self.domain_estimators[domain]

        # 获取不确定性估计
        uncertainty_output = estimator.forward_with_uncertainty(embeddings)

        # 提取各种不确定性度量
        mc_samples = uncertainty_output['mc_samples']
        prediction_mean = uncertainty_output['prediction_mean']
        prediction_var = uncertainty_output['prediction_variance']

        # 计算不同类型的不确定性
        epistemic_uncertainty = self.compute_epistemic_uncertainty(mc_samples)
        aleatoric_uncertainty = self.compute_aleatoric_uncertainty(uncertainty_output)
        predictive_entropy = self.compute_predictive_entropy(mc_samples)
        mutual_information = self.compute_mutual_information(mc_samples)

        # 综合不确定性分数
        composite_uncertainty = (
            self.config.entropy_weight * predictive_entropy +
            self.config.variance_weight * epistemic_uncertainty.mean(dim=-1) +
            0.3 * mutual_information
        )

        # 计算置信区间宽度作为额外的不确定性度量
        lower_bound, upper_bound = self.compute_confidence_interval(mc_samples)
        confidence_width = torch.mean(upper_bound - lower_bound, dim=-1)

        # 最终困难度分数
        difficulty_score = composite_uncertainty + 0.2 * confidence_width

        return {
            'difficulty_score': difficulty_score,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'predictive_entropy': predictive_entropy,
            'mutual_information': mutual_information,
            'confidence_width': confidence_width,
            'prediction_mean': prediction_mean,
            'prediction_variance': prediction_var
        }

    def adaptive_threshold_selection(self, difficulty_scores: Dict[str, torch.Tensor],
                                   domain_stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """自适应阈值选择"""
        thresholds = {}

        for domain, scores in difficulty_scores.items():
            if self.config.adaptive_threshold and hasattr(self, 'threshold_net'):
                # 计算统计特征
                stats = [
                    torch.mean(scores),
                    torch.std(scores),
                    torch.median(scores),
                    torch.quantile(scores, 0.25),
                    torch.quantile(scores, 0.75),
                    torch.min(scores),
                    torch.max(scores),
                    float(len(scores))
                ]

                stats_tensor = torch.tensor(stats, device=scores.device)

                # 使用神经网络预测阈值
                adaptive_threshold = self.threshold_net(stats_tensor)
                thresholds[domain] = 0.3 + 0.6 * adaptive_threshold.item()  # 阈值范围[0.3, 0.9]

            else:
                # 使用固定阈值
                thresholds[domain] = self.config.uncertainty_threshold

            # 更新困难度历史
            self.difficulty_history[domain].extend(scores.cpu().numpy().tolist())
            if len(self.difficulty_history[domain]) > self.config.calibration_samples:
                self.difficulty_history[domain] = self.difficulty_history[domain][-self.config.calibration_samples:]

        return thresholds

    def calibrate_uncertainty(self, domain: str) -> float:
        """基于历史数据校准不确定性阈值"""
        if domain not in self.difficulty_history or len(self.difficulty_history[domain]) < 100:
            return self.config.uncertainty_threshold

        history = np.array(self.difficulty_history[domain])

        # 使用分位数作为校准阈值
        calibrated_threshold = np.percentile(history, 70)  # 选择70%分位数

        self.calibrated_thresholds[domain] = calibrated_threshold
        return calibrated_threshold

    def detect_hard_samples(self, embeddings_dict: Dict[str, torch.Tensor],
                          labels_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """多域困难样本检测主函数"""

        all_difficulty_scores = {}
        all_uncertainty_info = {}
        hard_sample_indices = {}
        hard_sample_weights = {}

        # 对每个域分别检测
        for domain, embeddings in embeddings_dict.items():
            domain_labels = labels_dict.get(domain) if labels_dict else None

            # 单域检测
            domain_results = self.detect_hard_samples_single_domain(
                embeddings, domain, domain_labels
            )

            difficulty_scores = domain_results['difficulty_score']
            all_difficulty_scores[domain] = difficulty_scores
            all_uncertainty_info[domain] = domain_results

            # 校准阈值
            calibrated_threshold = self.calibrate_uncertainty(domain)

        # 自适应阈值选择
        domain_stats = {domain: torch.tensor([torch.mean(scores), torch.std(scores)])
                       for domain, scores in all_difficulty_scores.items()}
        adaptive_thresholds = self.adaptive_threshold_selection(all_difficulty_scores, domain_stats)

        # 选择困难样本
        for domain, difficulty_scores in all_difficulty_scores.items():
            threshold = adaptive_thresholds[domain]

            # 困难样本索引
            hard_indices = torch.where(difficulty_scores > threshold)[0]

            # 如果困难样本太少，选择top-k
            if len(hard_indices) < self.config.top_k_hard_samples:
                _, top_k_indices = torch.topk(difficulty_scores,
                                            min(self.config.top_k_hard_samples, len(difficulty_scores)))
                hard_indices = top_k_indices

            hard_sample_indices[domain] = hard_indices

            # 困难样本权重 (基于困难度分数)
            weights = torch.ones_like(difficulty_scores)
            weights[hard_indices] = 1.0 + difficulty_scores[hard_indices]  # 给困难样本更高权重
            hard_sample_weights[domain] = weights

        return {
            'hard_sample_indices': hard_sample_indices,
            'hard_sample_weights': hard_sample_weights,
            'difficulty_scores': all_difficulty_scores,
            'uncertainty_info': all_uncertainty_info,
            'adaptive_thresholds': adaptive_thresholds,
            'calibrated_thresholds': self.calibrated_thresholds.copy()
        }

    def get_uncertainty_metrics(self, embeddings_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """获取不确定性相关的评估指标"""
        results = self.detect_hard_samples(embeddings_dict)

        metrics = {}
        for domain in embeddings_dict.keys():
            difficulty_scores = results['difficulty_scores'][domain]
            uncertainty_info = results['uncertainty_info'][domain]

            domain_metrics = {
                f'{domain}_avg_difficulty': torch.mean(difficulty_scores).item(),
                f'{domain}_std_difficulty': torch.std(difficulty_scores).item(),
                f'{domain}_max_difficulty': torch.max(difficulty_scores).item(),
                f'{domain}_min_difficulty': torch.min(difficulty_scores).item(),
                f'{domain}_hard_sample_ratio': len(results['hard_sample_indices'][domain]) / len(difficulty_scores),
                f'{domain}_avg_epistemic_uncertainty': torch.mean(uncertainty_info['epistemic_uncertainty']).item(),
                f'{domain}_avg_predictive_entropy': torch.mean(uncertainty_info['predictive_entropy']).item(),
                f'{domain}_avg_mutual_information': torch.mean(uncertainty_info['mutual_information']).item(),
            }

            metrics.update(domain_metrics)

        return metrics

def create_uncertainty_detector(config: Optional[UncertaintyConfig] = None) -> UncertaintyHardSampleDetector:
    """创建不确定性困难样本检测器"""
    if config is None:
        config = UncertaintyConfig()

    return UncertaintyHardSampleDetector(config)

# 使用示例和测试
if __name__ == "__main__":
    # 创建配置
    config = UncertaintyConfig(
        num_mc_samples=20,
        dropout_rate=0.1,
        uncertainty_threshold=0.7,
        adaptive_threshold=True,
        top_k_hard_samples=16
    )

    # 创建检测器
    detector = create_uncertainty_detector(config)

    # 模拟多域嵌入数据
    batch_size = 64
    embeddings_dict = {
        'vision': torch.randn(batch_size, 256),
        'nlp': torch.randn(batch_size, 256),
        'security': torch.randn(batch_size, 256),
        'medical': torch.randn(batch_size, 256)
    }

    # 模拟标签
    labels_dict = {
        'vision': torch.randint(0, 10, (batch_size,)),
        'nlp': torch.randint(0, 2, (batch_size,)),
        'security': torch.randint(0, 2, (batch_size,)),
        'medical': torch.randint(0, 5, (batch_size,))
    }

    # 检测困难样本
    results = detector.detect_hard_samples(embeddings_dict, labels_dict)

    print("=== 基于不确定性的困难样本检测结果 ===")

    for domain in embeddings_dict.keys():
        hard_indices = results['hard_sample_indices'][domain]
        difficulty_scores = results['difficulty_scores'][domain]

        print(f"\n{domain.capitalize()} 域:")
        print(f"  困难样本数量: {len(hard_indices)}")
        print(f"  困难样本比例: {len(hard_indices)/len(difficulty_scores):.2%}")
        print(f"  平均困难度分数: {torch.mean(difficulty_scores).item():.4f}")
        print(f"  自适应阈值: {results['adaptive_thresholds'][domain]:.4f}")

    # 获取详细指标
    metrics = detector.get_uncertainty_metrics(embeddings_dict)
    print("\n=== 详细不确定性指标 ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")