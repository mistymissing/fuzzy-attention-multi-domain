#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniMatch-Clip 核心算法重构：自适应跨域对齐器
论文导向的高质量算法实现，具有严格的理论基础
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class AlignmentConfig:
    """对齐配置参数"""
    temperature: float = 0.07  # 对比学习温度参数
    margin: float = 0.1  # 对齐边距
    alpha: float = 0.5  # 域间权重平衡参数
    beta: float = 0.3  # 困难样本权重
    num_negative_samples: int = 128  # 负样本数量
    adaptive_temperature: bool = True  # 自适应温度调节
    uncertainty_threshold: float = 0.7  # 不确定性阈值

class DomainAwareContrastiveLoss(nn.Module):
    """域感知对比学习损失"""

    def __init__(self, config: AlignmentConfig):
        super().__init__()
        self.config = config
        self.temperature = nn.Parameter(torch.tensor(config.temperature))

        # 域间权重矩阵 (可学习)
        self.domain_weights = nn.Parameter(torch.eye(4))  # 4个域的权重矩阵

        # 自适应温度网络
        if config.adaptive_temperature:
            self.temp_net = nn.Sequential(
                nn.Linear(256, 64),  # 假设特征维度为256
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def compute_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算相似度矩阵"""
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算余弦相似度
        similarity_matrix = torch.matmul(embeddings, embeddings.t())

        return similarity_matrix

    def adaptive_temperature_scaling(self, embeddings: torch.Tensor,
                                  domain_ids: torch.Tensor) -> torch.Tensor:
        """自适应温度缩放"""
        if not self.config.adaptive_temperature:
            return self.temperature.expand(embeddings.size(0))

        # 基于嵌入特征计算自适应温度
        adaptive_temps = self.temp_net(embeddings).squeeze(-1)

        # 温度范围限制在[0.01, 1.0]
        adaptive_temps = 0.01 + 0.99 * adaptive_temps

        return adaptive_temps

    def domain_aware_sampling(self, similarity_matrix: torch.Tensor,
                           domain_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """域感知负样本采样"""
        batch_size = similarity_matrix.size(0)

        positive_pairs = []
        negative_pairs = []

        for i in range(batch_size):
            current_domain = domain_ids[i]

            # 正样本：同域中的其他样本
            same_domain_mask = (domain_ids == current_domain)
            same_domain_indices = torch.where(same_domain_mask)[0]
            same_domain_indices = same_domain_indices[same_domain_indices != i]

            if len(same_domain_indices) > 0:
                pos_idx = torch.randint(0, len(same_domain_indices), (1,))
                positive_pairs.append((i, same_domain_indices[pos_idx].item()))

            # 负样本：其他域的样本，基于困难度采样
            other_domain_mask = (domain_ids != current_domain)
            other_domain_indices = torch.where(other_domain_mask)[0]

            if len(other_domain_indices) > 0:
                # 基于相似度进行困难负样本挖掘
                similarities = similarity_matrix[i][other_domain_indices]

                # 选择相似度较高的困难负样本
                num_negatives = min(self.config.num_negative_samples, len(other_domain_indices))
                hard_negatives_idx = torch.topk(similarities, num_negatives, largest=True)[1]

                for neg_idx in hard_negatives_idx:
                    negative_pairs.append((i, other_domain_indices[neg_idx].item()))

        return positive_pairs, negative_pairs

    def forward(self, embeddings_dict: Dict[str, torch.Tensor],
                domain_ids: torch.Tensor,
                hard_sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""

        # 合并所有域的嵌入
        all_embeddings = []
        all_domain_ids = []

        domain_name_to_id = {'vision': 0, 'nlp': 1, 'security': 2, 'medical': 3}

        for domain_name, embeddings in embeddings_dict.items():
            all_embeddings.append(embeddings)
            domain_id = domain_name_to_id[domain_name]
            all_domain_ids.extend([domain_id] * embeddings.size(0))

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_domain_ids = torch.tensor(all_domain_ids, device=all_embeddings.device)

        # 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(all_embeddings)

        # 自适应温度缩放
        temperatures = self.adaptive_temperature_scaling(all_embeddings, all_domain_ids)

        # 域感知采样
        positive_pairs, negative_pairs = self.domain_aware_sampling(similarity_matrix, all_domain_ids)

        # 计算对比学习损失
        total_loss = 0.0
        num_pairs = 0

        # 正样本对损失
        for anchor_idx, pos_idx in positive_pairs:
            anchor_temp = temperatures[anchor_idx]
            pos_similarity = similarity_matrix[anchor_idx, pos_idx] / anchor_temp

            # 计算与所有负样本的对比
            neg_similarities = []
            for neg_anchor_idx, neg_idx in negative_pairs:
                if neg_anchor_idx == anchor_idx:
                    neg_sim = similarity_matrix[anchor_idx, neg_idx] / anchor_temp
                    neg_similarities.append(neg_sim)

            if neg_similarities:
                neg_similarities = torch.stack(neg_similarities)

                # InfoNCE损失
                logits = torch.cat([pos_similarity.unsqueeze(0), neg_similarities])
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)

                loss = F.cross_entropy(logits.unsqueeze(0), labels)

                # 应用困难样本权重
                if hard_sample_weights is not None:
                    weight = hard_sample_weights[anchor_idx]
                    loss = loss * weight

                total_loss += loss
                num_pairs += 1

        # 域间对齐正则化
        domain_alignment_loss = self.compute_domain_alignment_regularization(
            embeddings_dict, all_domain_ids
        )

        if num_pairs > 0:
            contrastive_loss = total_loss / num_pairs
            total_loss = contrastive_loss + self.config.alpha * domain_alignment_loss
        else:
            total_loss = self.config.alpha * domain_alignment_loss

        return total_loss

    def compute_domain_alignment_regularization(self, embeddings_dict: Dict[str, torch.Tensor],
                                              all_domain_ids: torch.Tensor) -> torch.Tensor:
        """计算域间对齐正则化项"""
        domains = list(embeddings_dict.keys())
        reg_loss = 0.0
        num_pairs = 0

        # 计算每个域的中心
        domain_centers = {}
        for domain_name, embeddings in embeddings_dict.items():
            if embeddings.size(0) > 0:
                domain_centers[domain_name] = torch.mean(embeddings, dim=0)

        # 域中心间的对齐损失
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains[i+1:], i+1):
                if domain1 in domain_centers and domain2 in domain_centers:
                    center1 = domain_centers[domain1]
                    center2 = domain_centers[domain2]

                    # 计算域中心间的距离
                    distance = torch.norm(center1 - center2, p=2)

                    # 使用可学习的域权重
                    weight = self.domain_weights[i, j]

                    reg_loss += weight * distance
                    num_pairs += 1

        return reg_loss / max(num_pairs, 1)

class AdaptiveCrossDomainAligner(nn.Module):
    """自适应跨域对齐器主类"""

    def __init__(self, config: AlignmentConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # 核心组件
        self.contrastive_loss = DomainAwareContrastiveLoss(config)

        # 跨域投影网络
        self.domain_projectors = nn.ModuleDict({
            'vision': nn.Sequential(
                nn.Linear(1000, 512),  # 假设ResNet输出1000维
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.L2Norm(dim=1)  # L2归一化
            ),
            'nlp': nn.Sequential(
                nn.Linear(768, 512),   # 假设BERT输出768维
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.L2Norm(dim=1)
            ),
            'security': nn.Sequential(
                nn.Linear(100, 256),   # 安全特征维度
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.L2Norm(dim=1)
            ),
            'medical': nn.Sequential(
                nn.Linear(1000, 512),  # 医疗图像特征
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.L2Norm(dim=1)
            )
        })

        # 对齐质量评估网络
        self.alignment_quality_net = nn.Sequential(
            nn.Linear(256 * 4, 512),  # 4个域的特征连接
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def project_embeddings(self, embeddings_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将不同域的嵌入投影到统一空间"""
        projected_embeddings = {}

        for domain, embeddings in embeddings_dict.items():
            if domain in self.domain_projectors:
                projected = self.domain_projectors[domain](embeddings)
                projected_embeddings[domain] = projected
            else:
                self.logger.warning(f"No projector found for domain: {domain}")
                projected_embeddings[domain] = embeddings

        return projected_embeddings

    def compute_cross_domain_similarity(self, embeddings_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算跨域相似度矩阵"""
        domains = list(embeddings_dict.keys())
        num_domains = len(domains)

        similarity_matrix = torch.zeros(num_domains, num_domains)

        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                if domain1 != domain2:
                    emb1 = embeddings_dict[domain1]
                    emb2 = embeddings_dict[domain2]

                    # 计算域间平均相似度
                    sim_matrix = torch.matmul(F.normalize(emb1, p=2, dim=1),
                                            F.normalize(emb2, p=2, dim=1).t())
                    avg_similarity = torch.mean(sim_matrix)
                    similarity_matrix[i, j] = avg_similarity
                else:
                    similarity_matrix[i, j] = 1.0  # 自己与自己的相似度为1

        return similarity_matrix

    def evaluate_alignment_quality(self, embeddings_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """评估对齐质量"""
        # 获取每个域的代表性特征（平均池化）
        domain_features = []
        for domain in ['vision', 'nlp', 'security', 'medical']:
            if domain in embeddings_dict and embeddings_dict[domain].size(0) > 0:
                avg_feature = torch.mean(embeddings_dict[domain], dim=0)
                domain_features.append(avg_feature)
            else:
                # 如果该域没有样本，使用零向量
                domain_features.append(torch.zeros(256, device=list(embeddings_dict.values())[0].device))

        # 连接所有域的特征
        concatenated_features = torch.cat(domain_features)

        # 预测对齐质量分数
        quality_score = self.alignment_quality_net(concatenated_features)

        return quality_score

    def forward(self, domain_outputs: Dict[str, torch.Tensor],
                hard_sample_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播"""

        # 1. 投影到统一空间
        projected_embeddings = self.project_embeddings(domain_outputs)

        # 2. 计算跨域相似度
        cross_domain_similarity = self.compute_cross_domain_similarity(projected_embeddings)

        # 3. 构造域标签
        domain_ids = []
        domain_name_to_id = {'vision': 0, 'nlp': 1, 'security': 2, 'medical': 3}

        for domain_name, embeddings in projected_embeddings.items():
            domain_id = domain_name_to_id[domain_name]
            domain_ids.extend([domain_id] * embeddings.size(0))

        domain_ids = torch.tensor(domain_ids, device=list(projected_embeddings.values())[0].device)

        # 4. 计算对齐损失
        alignment_loss = self.contrastive_loss(projected_embeddings, domain_ids, hard_sample_weights)

        # 5. 评估对齐质量
        alignment_quality = self.evaluate_alignment_quality(projected_embeddings)

        return {
            'aligned_embeddings': projected_embeddings,
            'alignment_loss': alignment_loss,
            'cross_domain_similarity': cross_domain_similarity,
            'alignment_quality': alignment_quality
        }

    def get_alignment_metrics(self, embeddings_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """获取对齐指标用于评估"""
        projected_embeddings = self.project_embeddings(embeddings_dict)
        similarity_matrix = self.compute_cross_domain_similarity(projected_embeddings)
        quality_score = self.evaluate_alignment_quality(projected_embeddings)

        # 计算各种对齐指标
        metrics = {
            'avg_cross_domain_similarity': torch.mean(similarity_matrix[similarity_matrix != 1.0]).item(),
            'alignment_quality_score': quality_score.item(),
            'vision_nlp_similarity': similarity_matrix[0, 1].item(),
            'vision_security_similarity': similarity_matrix[0, 2].item(),
            'vision_medical_similarity': similarity_matrix[0, 3].item(),
            'nlp_security_similarity': similarity_matrix[1, 2].item(),
            'nlp_medical_similarity': similarity_matrix[1, 3].item(),
            'security_medical_similarity': similarity_matrix[2, 3].item(),
        }

        return metrics

class L2Norm(nn.Module):
    """L2归一化层"""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

def create_adaptive_aligner(config: Optional[AlignmentConfig] = None) -> AdaptiveCrossDomainAligner:
    """创建自适应跨域对齐器"""
    if config is None:
        config = AlignmentConfig()

    return AdaptiveCrossDomainAligner(config)

# 使用示例和测试
if __name__ == "__main__":
    # 创建配置
    config = AlignmentConfig(
        temperature=0.07,
        margin=0.1,
        alpha=0.5,
        beta=0.3,
        adaptive_temperature=True
    )

    # 创建对齐器
    aligner = create_adaptive_aligner(config)

    # 模拟多域输入数据
    batch_size = 32
    domain_outputs = {
        'vision': torch.randn(batch_size, 1000),      # ResNet特征
        'nlp': torch.randn(batch_size, 768),          # BERT特征
        'security': torch.randn(batch_size, 100),     # 安全特征
        'medical': torch.randn(batch_size, 1000)      # 医疗图像特征
    }

    # 模拟困难样本权重
    total_samples = sum(emb.size(0) for emb in domain_outputs.values())
    hard_sample_weights = torch.rand(total_samples)

    # 前向传播
    results = aligner(domain_outputs, hard_sample_weights)

    print("=== 自适应跨域对齐器测试结果 ===")
    print(f"对齐损失: {results['alignment_loss'].item():.4f}")
    print(f"对齐质量分数: {results['alignment_quality'].item():.4f}")
    print("跨域相似度矩阵:")
    print(results['cross_domain_similarity'])

    # 获取详细指标
    metrics = aligner.get_alignment_metrics(domain_outputs)
    print("\n详细对齐指标:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")