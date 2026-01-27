# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# GPU并行测试配置
# --------------------------------------------------------

import os
import pytest
import subprocess
from typing import List


# 在conftest.py加载时尽早设置GPU配置
# 这段代码会在import torch之前执行
_worker_id = os.environ.get('PYTEST_XDIST_WORKER', None)

if _worker_id:
    # 这是worker进程，尽早设置CUDA_VISIBLE_DEVICES
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        gpu_config = os.environ.get('PYTEST_GPU_CONFIG', None)
        if gpu_config:
            free_gpus = [int(x) for x in gpu_config.split(',')]
            # 从worker ID提取编号 (例如: 'gw0' -> 0)
            if _worker_id.startswith('gw'):
                worker_num = int(_worker_id[2:])
                assigned_gpu = free_gpus[worker_num % len(free_gpus)]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(assigned_gpu)
                print(f"Worker {_worker_id} 被分配到GPU {assigned_gpu}")
else:
    # 这是主进程，设置GPU配置环境变量供worker使用
    if 'PYTEST_GPU_CONFIG' not in os.environ:
        # 检测可用GPU并设置到环境变量
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            free_gpus = []
            for line in result.stdout.strip().split('\n'):
                gpu_id, mem_used = line.split(',')
                gpu_id = int(gpu_id.strip())
                mem_used = float(mem_used.strip())
                if mem_used < 1000:  # 显存使用低于1GB
                    free_gpus.append(gpu_id)
            
            if free_gpus:
                os.environ['PYTEST_GPU_CONFIG'] = ','.join(map(str, free_gpus))
        except:
            pass  # 如果检测失败，不设置环境变量


def pytest_configure(config):
    """pytest配置钩子：在测试开始前执行"""
    # 检查是否是worker进程
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', None)
    
    if worker_id is None:
        # 这是主进程
        gpu_config = os.environ.get('PYTEST_GPU_CONFIG', '')
        if gpu_config:
            free_gpus = [int(x) for x in gpu_config.split(',')]
        else:
            free_gpus = []
        
        config.free_gpus = free_gpus
        
        # 如果使用xdist并行测试，显示GPU信息
        if config.getoption('dist', 'no') != 'no' and free_gpus:
            print(f"\n检测到{len(free_gpus)}个可用GPU: {free_gpus}")
            if config.getoption('numprocesses', None):
                num_workers = int(config.getoption('numprocesses'))
                print(f"将启动{num_workers}个并行worker")
    else:
        # Worker进程，不需要free_gpus
        config.free_gpus = []


@pytest.fixture(scope="session")
def gpu_device():
    """返回当前进程应该使用的GPU设备
    
    在并行测试时，每个worker会被分配到不同的GPU
    """
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    # 如果CUDA_VISIBLE_DEVICES被设置，PyTorch会将第一个可见设备映射为cuda:0
    return 'cuda'


@pytest.fixture(scope="function")
def cleanup_cuda():
    """测试后清理CUDA缓存"""
    yield
    # 测试完成后清理
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
