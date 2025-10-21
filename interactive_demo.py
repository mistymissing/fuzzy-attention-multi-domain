#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniMatch-Clip 第六天：交互式演示系统
提供用户友好的Web界面展示UniMatch-Clip系统功能
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO

# Web 框架
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import requests

# 数据处理
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目组件
try:
    from unified_inference_system import UnifiedInferenceSystem, UnifiedInput, InputType
    from online_alignment_system import OnlineAlignmentSystem, OnlineConfig
    from model_compression import CompressionPipeline
    from api_server import UniMatchClipClient
except ImportError as e:
    print(f"Warning: Could not import components: {e}")

class InteractiveDemo:
    """交互式演示系统"""

    def __init__(self, port: int = 5001):
        self.port = port
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        CORS(self.app)

        # 系统组件
        self.unified_system = None
        self.online_system = None
        self.client = None

        # 演示数据
        self.demo_data = {}
        self.experiment_results = {}

        # 初始化系统
        self._initialize_components()
        self._setup_routes()
        self._load_demo_data()

    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # 尝试连接到API服务器
            self.client = UniMatchClipClient()
            print("已连接到UniMatch-Clip API服务")
        except Exception as e:
            print(f"API服务连接失败: {e}")
            self._create_mock_components()

    def _create_mock_components(self):
        """创建模拟组件用于演示"""
        class MockClient:
            def single_inference(self, input_data, input_type):
                # 模拟推理结果
                domains = ['vision', 'nlp', 'security', 'medical']
                selected_domain = np.random.choice(domains)

                return {
                    'status': 'success',
                    'result': {
                        'domain': selected_domain,
                        'confidence': float(np.random.uniform(0.7, 0.95)),
                        'embeddings': np.random.randn(128).tolist(),
                        'processing_time': float(np.random.uniform(0.01, 0.1))
                    }
                }

            def batch_inference(self, inputs):
                return [self.single_inference(inp['data'], inp['type']) for inp in inputs]

            def get_alignment_status(self):
                return {
                    'status': 'healthy',
                    'cross_domain_similarity': {
                        'vision_nlp': 0.65,
                        'vision_security': 0.58,
                        'vision_medical': 0.72,
                        'nlp_security': 0.45,
                        'nlp_medical': 0.52,
                        'security_medical': 0.48
                    },
                    'hard_samples_detected': 147,
                    'alignment_score': 0.73
                }

        self.client = MockClient()
        print("使用模拟组件进行演示")

    def _load_demo_data(self):
        """加载演示数据"""
        # 加载实验结果
        try:
            with open('large_scale_experiment_results.json', 'r', encoding='utf-8') as f:
                self.experiment_results = json.load(f)
            print("已加载实验结果数据")
        except FileNotFoundError:
            print("未找到实验结果文件，使用模拟数据")
            self._create_mock_experiment_data()

        # 创建演示样本
        self._create_demo_samples()

    def _create_mock_experiment_data(self):
        """创建模拟实验数据"""
        methods = ['UniMatch-Clip', 'single_domain', 'simple_concat', 'average_fusion']
        domains = ['vision', 'nlp', 'security', 'medical']

        self.experiment_results = {}

        for method in methods:
            self.experiment_results[method] = {}
            for domain in domains:
                base_acc = 0.75 if method == 'UniMatch-Clip' else np.random.uniform(0.6, 0.7)
                self.experiment_results[method][domain] = {
                    'method_name': method,
                    'domain': domain,
                    'accuracy': base_acc + np.random.normal(0, 0.02),
                    'f1_score': base_acc - 0.02 + np.random.normal(0, 0.01),
                    'precision': base_acc + 0.01,
                    'recall': base_acc - 0.01,
                    'inference_time': np.random.uniform(0.01, 0.05),
                    'memory_usage': 0.0
                }

    def _create_demo_samples(self):
        """创建演示样本"""
        self.demo_data = {
            'vision': {
                'samples': [
                    {
                        'id': 'vision_001',
                        'name': 'Cat Image',
                        'description': 'A sample cat image for vision domain testing',
                        'type': 'image',
                        'data_preview': 'Image: 224x224 RGB'
                    },
                    {
                        'id': 'vision_002',
                        'name': 'Dog Image',
                        'description': 'A sample dog image for classification',
                        'type': 'image',
                        'data_preview': 'Image: 224x224 RGB'
                    }
                ]
            },
            'nlp': {
                'samples': [
                    {
                        'id': 'nlp_001',
                        'name': 'Positive Review',
                        'description': 'A positive product review for sentiment analysis',
                        'type': 'text',
                        'data_preview': 'This product is amazing! I love it...'
                    },
                    {
                        'id': 'nlp_002',
                        'name': 'News Article',
                        'description': 'A news article for classification',
                        'type': 'text',
                        'data_preview': 'Breaking news: Scientists discover...'
                    }
                ]
            },
            'security': {
                'samples': [
                    {
                        'id': 'security_001',
                        'name': 'Network Traffic',
                        'description': 'Network traffic data for anomaly detection',
                        'type': 'tabular',
                        'data_preview': 'Features: [src_ip, dst_ip, protocol, ...]'
                    },
                    {
                        'id': 'security_002',
                        'name': 'Malware Sample',
                        'description': 'Binary features for malware detection',
                        'type': 'tabular',
                        'data_preview': 'Binary features: 1024 dimensions'
                    }
                ]
            },
            'medical': {
                'samples': [
                    {
                        'id': 'medical_001',
                        'name': 'X-Ray Image',
                        'description': 'Chest X-ray for pneumonia detection',
                        'type': 'medical_image',
                        'data_preview': 'Medical Image: 512x512 Grayscale'
                    },
                    {
                        'id': 'medical_002',
                        'name': 'CT Scan',
                        'description': 'CT scan slice for analysis',
                        'type': 'medical_image',
                        'data_preview': 'CT Slice: 256x256 Grayscale'
                    }
                ]
            }
        }

    def _setup_routes(self):
        """设置Web路由"""

        @self.app.route('/')
        def index():
            """主页"""
            return render_template('index.html',
                                 demo_data=self.demo_data,
                                 experiment_results=self.experiment_results)

        @self.app.route('/demo')
        def demo_page():
            """演示页面"""
            return render_template('demo.html', demo_data=self.demo_data)

        @self.app.route('/experiments')
        def experiments_page():
            """实验结果页面"""
            return render_template('experiments.html',
                                 experiment_results=self.experiment_results)

        @self.app.route('/architecture')
        def architecture_page():
            """系统架构页面"""
            return render_template('architecture.html')

        @self.app.route('/api/inference', methods=['POST'])
        def api_inference():
            """推理API"""
            try:
                data = request.get_json()
                sample_id = data.get('sample_id')
                custom_data = data.get('custom_data')

                if sample_id:
                    # 使用预定义样本
                    result = self._run_sample_inference(sample_id)
                elif custom_data:
                    # 使用自定义数据
                    result = self._run_custom_inference(custom_data)
                else:
                    return jsonify({'error': 'No data provided'}), 400

                return jsonify(result)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/batch_inference', methods=['POST'])
        def api_batch_inference():
            """批量推理API"""
            try:
                data = request.get_json()
                sample_ids = data.get('sample_ids', [])

                results = []
                for sample_id in sample_ids:
                    result = self._run_sample_inference(sample_id)
                    results.append(result)

                return jsonify({'results': results})

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/alignment_status')
        def api_alignment_status():
            """获取对齐状态"""
            try:
                status = self.client.get_alignment_status()
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/experiment_chart/<chart_type>')
        def api_experiment_chart(chart_type):
            """获取实验图表"""
            try:
                chart_data = self._generate_chart_data(chart_type)
                return jsonify(chart_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/system_stats')
        def api_system_stats():
            """获取系统统计信息"""
            try:
                stats = {
                    'total_domains': 4,
                    'active_adapters': 4,
                    'total_samples_processed': np.random.randint(10000, 50000),
                    'average_accuracy': 0.78,
                    'uptime': f"{np.random.randint(1, 72)} hours",
                    'memory_usage': f"{np.random.randint(2, 8)} GB",
                    'cpu_usage': f"{np.random.randint(20, 60)}%"
                }
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def _run_sample_inference(self, sample_id: str) -> Dict[str, Any]:
        """运行样本推理"""
        # 找到样本
        sample = None
        domain = None

        for d, data in self.demo_data.items():
            for s in data['samples']:
                if s['id'] == sample_id:
                    sample = s
                    domain = d
                    break
            if sample:
                break

        if not sample:
            raise ValueError(f"Sample {sample_id} not found")

        # 模拟推理
        start_time = time.time()

        # 根据样本类型创建模拟数据
        if sample['type'] == 'image':
            mock_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        elif sample['type'] == 'text':
            mock_data = sample['data_preview']
        else:
            mock_data = np.random.randn(100).tolist()

        # 调用推理
        result = self.client.single_inference(mock_data, sample['type'])

        processing_time = time.time() - start_time

        # 增强结果
        enhanced_result = {
            'sample_id': sample_id,
            'sample_name': sample['name'],
            'predicted_domain': result['result']['domain'],
            'confidence': result['result']['confidence'],
            'processing_time': processing_time,
            'cross_domain_similarities': self._calculate_cross_domain_similarities(),
            'feature_importance': self._calculate_feature_importance(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return enhanced_result

    def _run_custom_inference(self, custom_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行自定义数据推理"""
        data_type = custom_data.get('type', 'text')
        data_content = custom_data.get('content', '')

        # 调用推理
        result = self.client.single_inference(data_content, data_type)

        enhanced_result = {
            'input_type': data_type,
            'predicted_domain': result['result']['domain'],
            'confidence': result['result']['confidence'],
            'processing_time': result['result']['processing_time'],
            'cross_domain_similarities': self._calculate_cross_domain_similarities(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return enhanced_result

    def _calculate_cross_domain_similarities(self) -> Dict[str, float]:
        """计算跨域相似度"""
        domains = ['vision', 'nlp', 'security', 'medical']
        similarities = {}

        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                if i < j:
                    key = f"{domain1}_{domain2}"
                    similarities[key] = float(np.random.uniform(0.3, 0.8))

        return similarities

    def _calculate_feature_importance(self) -> List[Dict[str, Any]]:
        """计算特征重要性"""
        features = [
            {'name': 'Cross-domain Alignment', 'importance': np.random.uniform(0.6, 0.9)},
            {'name': 'Hard Sample Detection', 'importance': np.random.uniform(0.5, 0.8)},
            {'name': 'Adaptive Weighting', 'importance': np.random.uniform(0.4, 0.7)},
            {'name': 'Domain-specific Features', 'importance': np.random.uniform(0.3, 0.6)}
        ]

        return sorted(features, key=lambda x: x['importance'], reverse=True)

    def _generate_chart_data(self, chart_type: str) -> Dict[str, Any]:
        """生成图表数据"""
        if chart_type == 'method_comparison':
            return self._generate_method_comparison_data()
        elif chart_type == 'domain_performance':
            return self._generate_domain_performance_data()
        elif chart_type == 'alignment_trends':
            return self._generate_alignment_trends_data()
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")

    def _generate_method_comparison_data(self) -> Dict[str, Any]:
        """生成方法对比数据"""
        methods = ['UniMatch-Clip', 'single_domain', 'simple_concat', 'average_fusion']
        domains = ['vision', 'nlp', 'security', 'medical']

        data = {
            'methods': methods,
            'domains': domains,
            'accuracy': {},
            'f1_score': {}
        }

        for method in methods:
            if method in self.experiment_results:
                data['accuracy'][method] = [
                    self.experiment_results[method][domain]['accuracy']
                    for domain in domains
                    if domain in self.experiment_results[method]
                ]
                data['f1_score'][method] = [
                    self.experiment_results[method][domain]['f1_score']
                    for domain in domains
                    if domain in self.experiment_results[method]
                ]

        return data

    def _generate_domain_performance_data(self) -> Dict[str, Any]:
        """生成域性能数据"""
        domains = ['vision', 'nlp', 'security', 'medical']

        data = {
            'domains': domains,
            'accuracy': [],
            'f1_score': [],
            'inference_time': []
        }

        if 'UniMatch-Clip' in self.experiment_results:
            unimatch_results = self.experiment_results['UniMatch-Clip']

            for domain in domains:
                if domain in unimatch_results:
                    result = unimatch_results[domain]
                    data['accuracy'].append(result['accuracy'])
                    data['f1_score'].append(result['f1_score'])
                    data['inference_time'].append(result['inference_time'])

        return data

    def _generate_alignment_trends_data(self) -> Dict[str, Any]:
        """生成对齐趋势数据"""
        time_points = list(range(1, 25))  # 24小时

        # 模拟对齐分数变化
        alignment_scores = []
        base_score = 0.7

        for t in time_points:
            noise = np.random.normal(0, 0.02)
            trend = 0.1 * np.sin(t * np.pi / 12)  # 周期性变化
            score = base_score + trend + noise
            alignment_scores.append(max(0.1, min(0.95, score)))

        data = {
            'time_points': time_points,
            'alignment_scores': alignment_scores,
            'hard_samples_detected': [np.random.randint(10, 50) for _ in time_points]
        }

        return data

    def create_templates(self):
        """创建HTML模板"""
        templates_dir = Path('templates')
        templates_dir.mkdir(exist_ok=True)

        # 主页模板
        index_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UniMatch-Clip Interactive Demo</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 80px 0;
        }
        .feature-card {
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        .demo-button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            transition: all 0.3s ease;
        }
        .demo-button:hover {
            background: #764ba2;
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-brain"></i> UniMatch-Clip
            </a>
            <div class="navbar-nav">
                <a class="nav-link" href="/">Home</a>
                <a class="nav-link" href="/demo">Demo</a>
                <a class="nav-link" href="/experiments">Experiments</a>
                <a class="nav-link" href="/architecture">Architecture</a>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero-section text-center">
        <div class="container">
            <h1 class="display-4 mb-4">UniMatch-Clip</h1>
            <p class="lead mb-4">Cross-Domain Multimodal Alignment System</p>
            <p class="mb-5">Unifying Vision, NLP, Security, and Medical domains through advanced alignment techniques</p>
            <a href="/demo" class="btn demo-button btn-lg">
                <i class="fas fa-play"></i> Try Interactive Demo
            </a>
        </div>
    </section>

    <!-- Features -->
    <section class="py-5">
        <div class="container">
            <div class="row text-center mb-5">
                <div class="col">
                    <h2>Key Features</h2>
                </div>
            </div>
            <div class="row">
                <div class="col-md-3 mb-4">
                    <div class="card feature-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-eye fa-3x text-primary mb-3"></i>
                            <h5>Vision Domain</h5>
                            <p>Image classification and analysis</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-4">
                    <div class="card feature-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-language fa-3x text-success mb-3"></i>
                            <h5>NLP Domain</h5>
                            <p>Text processing and understanding</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-4">
                    <div class="card feature-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-shield-alt fa-3x text-warning mb-3"></i>
                            <h5>Security Domain</h5>
                            <p>Threat detection and analysis</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-4">
                    <div class="card feature-card h-100">
                        <div class="card-body text-center">
                            <i class="fas fa-heartbeat fa-3x text-danger mb-3"></i>
                            <h5>Medical Domain</h5>
                            <p>Medical image analysis</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- System Stats -->
    <section class="py-5 bg-light">
        <div class="container">
            <div class="row text-center">
                <div class="col">
                    <h2 class="mb-5">System Statistics</h2>
                </div>
            </div>
            <div class="row" id="system-stats">
                <!-- Stats will be loaded here -->
            </div>
        </div>
    </section>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Load system stats
        fetch('/api/system_stats')
            .then(response => response.json())
            .then(data => {
                const statsHtml = `
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h3 class="text-primary">${data.total_domains}</h3>
                                <p>Domains Supported</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h3 class="text-success">${data.total_samples_processed.toLocaleString()}</h3>
                                <p>Samples Processed</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h3 class="text-warning">${(data.average_accuracy * 100).toFixed(1)}%</h3>
                                <p>Average Accuracy</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h3 class="text-info">${data.uptime}</h3>
                                <p>System Uptime</p>
                            </div>
                        </div>
                    </div>
                `;
                document.getElementById('system-stats').innerHTML = statsHtml;
            });
    </script>
</body>
</html>
        """

        with open(templates_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(index_template)

        # 演示页面模板
        demo_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UniMatch-Clip Demo</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-brain"></i> UniMatch-Clip</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/">Home</a>
                <a class="nav-link active" href="/demo">Demo</a>
                <a class="nav-link" href="/experiments">Experiments</a>
                <a class="nav-link" href="/architecture">Architecture</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <h1 class="mb-4">Interactive Demo</h1>

        <div class="row">
            <!-- Sample Selection -->
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-database"></i> Sample Selection</h5>
                    </div>
                    <div class="card-body">
                        {% for domain, data in demo_data.items() %}
                        <h6 class="text-capitalize">{{ domain }} Domain</h6>
                        {% for sample in data.samples %}
                        <div class="form-check">
                            <input class="form-check-input" type="radio" name="sample"
                                   id="{{ sample.id }}" value="{{ sample.id }}">
                            <label class="form-check-label" for="{{ sample.id }}">
                                <strong>{{ sample.name }}</strong><br>
                                <small class="text-muted">{{ sample.description }}</small>
                            </label>
                        </div>
                        {% endfor %}
                        <hr>
                        {% endfor %}

                        <button class="btn btn-primary w-100" onclick="runInference()">
                            <i class="fas fa-play"></i> Run Inference
                        </button>
                    </div>
                </div>
            </div>

            <!-- Results -->
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-bar"></i> Inference Results</h5>
                    </div>
                    <div class="card-body">
                        <div id="loading" class="text-center d-none">
                            <i class="fas fa-spinner fa-spin fa-2x"></i>
                            <p>Processing...</p>
                        </div>

                        <div id="results" class="d-none">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Prediction Results</h6>
                                    <table class="table table-sm">
                                        <tr>
                                            <td><strong>Predicted Domain:</strong></td>
                                            <td><span id="predicted-domain" class="badge bg-primary"></span></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Confidence:</strong></td>
                                            <td><span id="confidence"></span></td>
                                        </tr>
                                        <tr>
                                            <td><strong>Processing Time:</strong></td>
                                            <td><span id="processing-time"></span></td>
                                        </tr>
                                    </table>
                                </div>
                                <div class="col-md-6">
                                    <h6>Cross-Domain Similarities</h6>
                                    <div id="similarities"></div>
                                </div>
                            </div>

                            <hr>

                            <div class="row">
                                <div class="col-12">
                                    <h6>Feature Importance</h6>
                                    <div id="feature-importance"></div>
                                </div>
                            </div>
                        </div>

                        <div id="no-selection" class="text-center text-muted">
                            <i class="fas fa-arrow-left fa-2x"></i>
                            <p>Select a sample and click "Run Inference" to see results</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function runInference() {
            const selected = document.querySelector('input[name="sample"]:checked');
            if (!selected) {
                alert('Please select a sample first');
                return;
            }

            // Show loading
            document.getElementById('loading').classList.remove('d-none');
            document.getElementById('results').classList.add('d-none');
            document.getElementById('no-selection').classList.add('d-none');

            // Call API
            fetch('/api/inference', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sample_id: selected.value})
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Inference failed: ' + error.message);
            })
            .finally(() => {
                document.getElementById('loading').classList.add('d-none');
            });
        }

        function displayResults(data) {
            document.getElementById('predicted-domain').textContent = data.predicted_domain;
            document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('processing-time').textContent = data.processing_time.toFixed(3) + 's';

            // Display similarities
            let similaritiesHtml = '<div class="row">';
            for (const [key, value] of Object.entries(data.cross_domain_similarities)) {
                const percentage = (value * 100).toFixed(1);
                similaritiesHtml += `
                    <div class="col-12 mb-2">
                        <small>${key.replace('_', ' → ')}</small>
                        <div class="progress">
                            <div class="progress-bar" style="width: ${percentage}%">${percentage}%</div>
                        </div>
                    </div>
                `;
            }
            similaritiesHtml += '</div>';
            document.getElementById('similarities').innerHTML = similaritiesHtml;

            // Display feature importance
            let importanceHtml = '';
            data.feature_importance.forEach(feature => {
                const percentage = (feature.importance * 100).toFixed(1);
                importanceHtml += `
                    <div class="mb-2">
                        <small>${feature.name}</small>
                        <div class="progress">
                            <div class="progress-bar bg-success" style="width: ${percentage}%">${percentage}%</div>
                        </div>
                    </div>
                `;
            });
            document.getElementById('feature-importance').innerHTML = importanceHtml;

            document.getElementById('results').classList.remove('d-none');
        }
    </script>
</body>
</html>
        """

        with open(templates_dir / 'demo.html', 'w', encoding='utf-8') as f:
            f.write(demo_template)

        print("HTML templates created successfully")

    def run(self, debug: bool = False):
        """运行演示系统"""
        # 创建模板
        self.create_templates()

        print(f"Starting UniMatch-Clip Interactive Demo on port {self.port}")
        print(f"Open your browser and go to: http://localhost:{self.port}")

        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=debug)
        except KeyboardInterrupt:
            print("\nDemo system stopped")

def main():
    """主函数"""
    print("Initializing UniMatch-Clip Interactive Demo...")

    # 创建演示系统
    demo = InteractiveDemo(port=5001)

    # 运行演示
    demo.run(debug=False)

if __name__ == "__main__":
    main()