#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniMatch-Clip 第五天：API服务接口
实现基于Flask的RESTful API服务，支持多域推理和实时对齐
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import torch
import numpy as np
import base64
import io
import json
import time
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 导入UniMatch-Clip组件
try:
    from unified_inference_system import UnifiedInferenceSystem, UnifiedInput, InputType
    from online_alignment_system import OnlineAlignmentSystem, OnlineConfig
    from model_compression import CompressionPipeline, CompressionConfig
except ImportError as e:
    print(f"Warning: Could not import UniMatch-Clip components: {e}")
    print("Running in mock mode...")

@dataclass
class APIResponse:
    """API响应格式"""
    success: bool
    data: Any = None
    error: str = None
    timestamp: str = None
    processing_time: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)

class APIRateLimiter:
    """API速率限制器"""

    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = {}
        self.lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        """检查是否允许请求"""
        current_time = time.time()
        minute_window = int(current_time / 60)

        with self.lock:
            if client_ip not in self.requests:
                self.requests[client_ip] = {}

            # 清理旧的时间窗口
            self.requests[client_ip] = {
                k: v for k, v in self.requests[client_ip].items()
                if k >= minute_window - 1
            }

            # 计算当前分钟的请求数
            current_minute_requests = self.requests[client_ip].get(minute_window, 0)

            if current_minute_requests >= self.max_requests:
                return False

            # 增加请求计数
            self.requests[client_ip][minute_window] = current_minute_requests + 1
            return True

def rate_limit(max_requests_per_minute: int = 60):
    """速率限制装饰器"""
    limiter = APIRateLimiter(max_requests_per_minute)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)

            if not limiter.is_allowed(client_ip):
                response = APIResponse(
                    success=False,
                    error="Rate limit exceeded. Please try again later."
                )
                return jsonify(response.to_dict()), 429

            return func(*args, **kwargs)
        return wrapper
    return decorator

def handle_errors(func):
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                response_data, status_code = result
            else:
                response_data, status_code = result, 200

            # 添加处理时间
            if isinstance(response_data, dict) and 'processing_time' not in response_data:
                response_data['processing_time'] = time.time() - start_time

            return response_data, status_code

        except Exception as e:
            logging.error(f"API Error in {func.__name__}: {str(e)}")
            logging.error(traceback.format_exc())

            response = APIResponse(
                success=False,
                error=f"Internal server error: {str(e)}",
                processing_time=time.time() - start_time
            )
            return jsonify(response.to_dict()), 500

    return wrapper

class UniMatchClipAPI:
    """UniMatch-Clip API服务"""

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # 启用跨域支持

        # 系统组件
        self.inference_system = None
        self.online_system = None
        self.compression_pipeline = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 统计信息
        self.request_count = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()

        # 配置日志
        self._setup_logging()

        # 初始化系统组件
        self._initialize_systems()

        # 注册路由
        self._register_routes()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('UniMatchClipAPI')

    def _initialize_systems(self):
        """初始化系统组件"""
        try:
            self.logger.info("正在初始化UniMatch-Clip系统组件...")

            # 初始化推理系统
            self.inference_system = UnifiedInferenceSystem()
            self.logger.info("统一推理系统初始化完成")

            # 初始化在线对齐系统
            config = OnlineConfig(
                window_size=50,
                update_frequency=10,
                confidence_threshold=0.6
            )
            self.online_system = OnlineAlignmentSystem(self.inference_system, config)
            self.logger.info("在线对齐系统初始化完成")

            # 初始化压缩流水线
            compression_config = CompressionConfig(
                enable_dynamic_quantization=True,
                enable_static_quantization=False
            )
            self.compression_pipeline = CompressionPipeline(compression_config)
            self.logger.info("模型压缩流水线初始化完成")

        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            # 创建模拟系统用于测试
            self._create_mock_systems()

    def _create_mock_systems(self):
        """创建模拟系统"""
        class MockSystem:
            def __init__(self):
                pass

            def preprocess_input(self, data, input_type):
                return type('MockInput', (), {
                    'data': data,
                    'input_type': input_type,
                    'domain': 'mock',
                    'metadata': {}
                })()

            def single_inference(self, unified_input):
                return type('MockOutput', (), {
                    'embeddings': torch.randn(1, 128),
                    'domain': unified_input.domain,
                    'confidence': 0.85,
                    'processing_time': 0.05,
                    'metadata': {}
                })()

            def batch_inference(self, inputs):
                outputs = [self.single_inference(inp) for inp in inputs]
                return type('BatchResult', (), {
                    'outputs': outputs,
                    'total_processing_time': 0.1,
                    'cross_domain_similarities': {'mock-mock': 0.7}
                })()

        self.inference_system = MockSystem()
        self.online_system = None
        self.compression_pipeline = None
        self.logger.info("已创建模拟系统")

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/', methods=['GET'])
        def root():
            return jsonify({
                "service": "UniMatch-Clip API",
                "version": "1.0.0",
                "status": "running",
                "endpoints": [
                    "/health",
                    "/inference/single",
                    "/inference/batch",
                    "/alignment/status",
                    "/models/compress",
                    "/stats"
                ]
            })

        @self.app.route('/health', methods=['GET'])
        @handle_errors
        def health_check():
            """健康检查"""
            uptime = time.time() - self.start_time
            system_status = {
                "inference_system": self.inference_system is not None,
                "online_alignment": self.online_system is not None,
                "compression_pipeline": self.compression_pipeline is not None,
                "uptime_seconds": uptime,
                "request_count": self.request_count,
                "average_processing_time": self.total_processing_time / max(self.request_count, 1)
            }

            if self.online_system:
                health_info = self.online_system.get_system_health()
                system_status["alignment_health"] = health_info

            response = APIResponse(
                success=True,
                data=system_status
            )
            return jsonify(response.to_dict())

        @self.app.route('/inference/single', methods=['POST'])
        @rate_limit(max_requests_per_minute=120)
        @handle_errors
        def single_inference():
            """单样本推理"""
            data = request.get_json()
            if not data:
                response = APIResponse(
                    success=False,
                    error="No JSON data provided"
                )
                return jsonify(response.to_dict()), 400

            # 解析输入
            input_data = data.get('data')
            input_type_str = data.get('type', 'auto_detect')

            if input_data is None:
                response = APIResponse(
                    success=False,
                    error="Missing 'data' field"
                )
                return jsonify(response.to_dict()), 400

            # 转换输入类型
            try:
                if input_type_str == 'text':
                    input_type = InputType.TEXT
                elif input_type_str == 'image':
                    input_type = InputType.IMAGE
                    # 如果是base64编码的图像，解码
                    if isinstance(input_data, str) and input_data.startswith('data:image'):
                        input_data = self._decode_base64_image(input_data)
                elif input_type_str == 'tabular':
                    input_type = InputType.TABULAR
                    input_data = torch.tensor(input_data, dtype=torch.float32)
                elif input_type_str == 'medical_image':
                    input_type = InputType.MEDICAL_IMAGE
                    if isinstance(input_data, str) and input_data.startswith('data:image'):
                        input_data = self._decode_base64_image(input_data)
                else:
                    input_type = InputType.AUTO_DETECT
            except Exception as e:
                response = APIResponse(
                    success=False,
                    error=f"Invalid input data: {str(e)}"
                )
                return jsonify(response.to_dict()), 400

            # 执行推理
            unified_input = self.inference_system.preprocess_input(input_data, input_type)
            output = self.inference_system.single_inference(unified_input)

            # 更新统计
            self.request_count += 1
            self.total_processing_time += output.processing_time

            # 构建响应
            result_data = {
                "domain": output.domain,
                "confidence": output.confidence,
                "embeddings_shape": list(output.embeddings.shape) if torch.is_tensor(output.embeddings) else [],
                "processing_time": output.processing_time,
                "metadata": output.metadata or {}
            }

            # 如果请求包含嵌入，添加嵌入数据
            if data.get('include_embeddings', False):
                result_data["embeddings"] = output.embeddings.tolist() if torch.is_tensor(output.embeddings) else []

            response = APIResponse(
                success=True,
                data=result_data
            )
            return jsonify(response.to_dict())

        @self.app.route('/inference/batch', methods=['POST'])
        @rate_limit(max_requests_per_minute=30)
        @handle_errors
        def batch_inference():
            """批量推理"""
            data = request.get_json()
            if not data:
                response = APIResponse(
                    success=False,
                    error="No JSON data provided"
                )
                return jsonify(response.to_dict()), 400

            batch_data = data.get('batch', [])
            if not batch_data or len(batch_data) > 32:  # 限制批量大小
                response = APIResponse(
                    success=False,
                    error="Invalid batch size (must be 1-32)"
                )
                return jsonify(response.to_dict()), 400

            # 预处理批量输入
            unified_inputs = []
            for item in batch_data:
                try:
                    input_data = item.get('data')
                    input_type_str = item.get('type', 'auto_detect')

                    # 转换输入类型（同单推理逻辑）
                    if input_type_str == 'text':
                        input_type = InputType.TEXT
                    elif input_type_str == 'image':
                        input_type = InputType.IMAGE
                    elif input_type_str == 'tabular':
                        input_type = InputType.TABULAR
                        input_data = torch.tensor(input_data, dtype=torch.float32)
                    else:
                        input_type = InputType.AUTO_DETECT

                    unified_input = self.inference_system.preprocess_input(input_data, input_type)
                    unified_inputs.append(unified_input)

                except Exception as e:
                    response = APIResponse(
                        success=False,
                        error=f"Error processing batch item: {str(e)}"
                    )
                    return jsonify(response.to_dict()), 400

            # 执行批量推理
            batch_result = self.inference_system.batch_inference(unified_inputs)

            # 在线处理（如果可用）
            if self.online_system:
                online_result = self.online_system.process_online_batch(batch_result)

            # 更新统计
            self.request_count += len(batch_data)
            self.total_processing_time += batch_result.total_processing_time

            # 构建响应
            results = []
            for output in batch_result.outputs:
                result_data = {
                    "domain": output.domain,
                    "confidence": output.confidence,
                    "embeddings_shape": list(output.embeddings.shape) if torch.is_tensor(output.embeddings) else [],
                    "processing_time": output.processing_time
                }
                results.append(result_data)

            response_data = {
                "results": results,
                "total_processing_time": batch_result.total_processing_time,
                "cross_domain_similarities": batch_result.cross_domain_similarities or {},
                "batch_size": len(batch_data)
            }

            response = APIResponse(
                success=True,
                data=response_data
            )
            return jsonify(response.to_dict())

        @self.app.route('/alignment/status', methods=['GET'])
        @handle_errors
        def alignment_status():
            """获取对齐系统状态"""
            if not self.online_system:
                response = APIResponse(
                    success=False,
                    error="Online alignment system not available"
                )
                return jsonify(response.to_dict()), 503

            health_info = self.online_system.get_system_health()

            response = APIResponse(
                success=True,
                data=health_info
            )
            return jsonify(response.to_dict())

        @self.app.route('/models/compress', methods=['POST'])
        @rate_limit(max_requests_per_minute=5)
        @handle_errors
        def compress_model():
            """模型压缩"""
            if not self.compression_pipeline:
                response = APIResponse(
                    success=False,
                    error="Model compression pipeline not available"
                )
                return jsonify(response.to_dict()), 503

            data = request.get_json()
            model_type = data.get('model_type', 'vision')

            if model_type not in ['vision', 'nlp', 'security', 'medical']:
                response = APIResponse(
                    success=False,
                    error="Invalid model type"
                )
                return jsonify(response.to_dict()), 400

            # 创建示例模型进行压缩演示
            try:
                if model_type == 'vision':
                    model = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 16, 3),
                        torch.nn.ReLU(),
                        torch.nn.AdaptiveAvgPool2d(1),
                        torch.nn.Flatten(),
                        torch.nn.Linear(16, 128)
                    )
                elif model_type == 'nlp':
                    model = torch.nn.Sequential(
                        torch.nn.Embedding(1000, 64),
                        torch.nn.Flatten(),
                        torch.nn.Linear(64*20, 128)
                    )
                else:  # security, medical
                    model = torch.nn.Sequential(
                        torch.nn.Linear(122, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 128)
                    )

                # 包装模型
                class WrappedModel(torch.nn.Module):
                    def __init__(self, base_model):
                        super().__init__()
                        self.base_model = base_model

                    def forward(self, x):
                        logits = self.base_model(x)
                        return {
                            'logits': logits,
                            'embeddings': logits,
                            'confidence_scores': torch.softmax(logits, dim=-1).max(dim=-1)[0]
                        }

                wrapped_model = WrappedModel(model)
                wrapped_model.eval()

                # 执行压缩
                compression_result = self.compression_pipeline.compress_model(wrapped_model, model_type)

                # 清理返回数据（移除PyTorch模型对象）
                clean_result = {}
                for key, value in compression_result.items():
                    if key != 'compressed_models' and not isinstance(value, torch.nn.Module):
                        clean_result[key] = value

                response = APIResponse(
                    success=True,
                    data=clean_result
                )
                return jsonify(response.to_dict())

            except Exception as e:
                response = APIResponse(
                    success=False,
                    error=f"Compression failed: {str(e)}"
                )
                return jsonify(response.to_dict()), 500

        @self.app.route('/stats', methods=['GET'])
        @handle_errors
        def get_stats():
            """获取API统计信息"""
            uptime = time.time() - self.start_time
            stats = {
                "uptime_seconds": uptime,
                "total_requests": self.request_count,
                "requests_per_minute": (self.request_count / uptime) * 60 if uptime > 0 else 0,
                "average_processing_time": self.total_processing_time / max(self.request_count, 1),
                "total_processing_time": self.total_processing_time,
                "system_components": {
                    "inference_system": self.inference_system is not None,
                    "online_alignment": self.online_system is not None,
                    "compression_pipeline": self.compression_pipeline is not None
                }
            }

            response = APIResponse(
                success=True,
                data=stats
            )
            return jsonify(response.to_dict())

    def _decode_base64_image(self, data_url: str) -> torch.Tensor:
        """解码base64图像"""
        # 简化实现：返回随机tensor
        # 实际实现应该解码base64图像并转换为tensor
        return torch.randn(1, 3, 224, 224)

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """运行API服务"""
        self.logger.info(f"Starting UniMatch-Clip API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

    def shutdown(self):
        """关闭服务"""
        if self.inference_system and hasattr(self.inference_system, 'shutdown'):
            self.inference_system.shutdown()
        if self.online_system:
            self.online_system.shutdown()
        self.executor.shutdown(wait=True)
        self.logger.info("API服务已关闭")

# API客户端示例
class UniMatchClipClient:
    """UniMatch-Clip API客户端"""

    def __init__(self, base_url: str = 'http://localhost:5000'):
        self.base_url = base_url
        self.session = None

    def _make_request(self, endpoint: str, method: str = 'GET', data: dict = None):
        """发送API请求"""
        import requests

        url = f"{self.base_url}{endpoint}"

        try:
            if method == 'GET':
                response = requests.get(url)
            elif method == 'POST':
                response = requests.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

    def health_check(self):
        """健康检查"""
        return self._make_request('/health')

    def single_inference(self, data: Any, data_type: str = 'auto_detect', include_embeddings: bool = False):
        """单样本推理"""
        request_data = {
            'data': data,
            'type': data_type,
            'include_embeddings': include_embeddings
        }
        return self._make_request('/inference/single', 'POST', request_data)

    def batch_inference(self, batch_data: List[Dict]):
        """批量推理"""
        request_data = {
            'batch': batch_data
        }
        return self._make_request('/inference/batch', 'POST', request_data)

    def get_alignment_status(self):
        """获取对齐状态"""
        return self._make_request('/alignment/status')

    def compress_model(self, model_type: str = 'vision'):
        """压缩模型"""
        request_data = {
            'model_type': model_type
        }
        return self._make_request('/models/compress', 'POST', request_data)

    def get_stats(self):
        """获取统计信息"""
        return self._make_request('/stats')

# 测试和演示
def demo_api_server():
    """API服务器演示"""
    print("启动UniMatch-Clip API服务器演示...")

    # 创建API服务器
    api_server = UniMatchClipAPI()

    # 在单独线程中运行服务器
    import threading
    server_thread = threading.Thread(target=lambda: api_server.run(debug=False))
    server_thread.daemon = True
    server_thread.start()

    # 等待服务器启动
    time.sleep(2)

    print("服务器已启动，正在测试API...")

    # 测试客户端
    client = UniMatchClipClient()

    # 健康检查
    print("\n1. 健康检查:")
    health_response = client.health_check()
    print(f"   状态: {health_response.get('success', False)}")

    # 单样本推理测试
    print("\n2. 单样本推理测试:")
    single_response = client.single_inference("This is a test sentence", "text")
    if single_response.get('success'):
        data = single_response.get('data', {})
        print(f"   域: {data.get('domain')}")
        print(f"   置信度: {data.get('confidence', 0):.3f}")
        print(f"   处理时间: {data.get('processing_time', 0):.4f}s")
    else:
        print(f"   错误: {single_response.get('error')}")

    # 批量推理测试
    print("\n3. 批量推理测试:")
    batch_data = [
        {"data": "Text sample 1", "type": "text"},
        {"data": "Text sample 2", "type": "text"},
        {"data": [1.0] * 122, "type": "tabular"}  # 安全特征
    ]
    batch_response = client.batch_inference(batch_data)
    if batch_response.get('success'):
        data = batch_response.get('data', {})
        print(f"   批量大小: {data.get('batch_size')}")
        print(f"   总处理时间: {data.get('total_processing_time', 0):.4f}s")
        print(f"   跨域相似度: {len(data.get('cross_domain_similarities', {}))}")
    else:
        print(f"   错误: {batch_response.get('error')}")

    # 统计信息
    print("\n4. 统计信息:")
    stats_response = client.get_stats()
    if stats_response.get('success'):
        data = stats_response.get('data', {})
        print(f"   总请求数: {data.get('total_requests', 0)}")
        print(f"   平均处理时间: {data.get('average_processing_time', 0):.4f}s")
        print(f"   运行时间: {data.get('uptime_seconds', 0):.1f}s")

    print("\nAPI演示完成!")

    # 关闭服务器
    api_server.shutdown()

if __name__ == "__main__":
    # 运行演示
    demo_api_server()