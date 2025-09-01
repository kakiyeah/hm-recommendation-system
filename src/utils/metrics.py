"""
评估指标模块

包含推荐系统常用的评估指标
"""

import numpy as np
from typing import List


def apk(actual: List, predicted: List, k: int = 12) -> float:
    """
    计算平均精度@k
    
    Args:
        actual: 真实标签列表
        predicted: 预测标签列表
        k: 截断位置
        
    Returns:
        平均精度@k
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    if not actual:
        return 0.0
    
    return score / min(len(actual), k)


def mapk(actual: List[List], predicted: List[List], k: int = 12) -> float:
    """
    计算平均精度@k
    
    Args:
        actual: 真实标签列表的列表
        predicted: 预测标签列表的列表
        k: 截断位置
        
    Returns:
        平均精度@k
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def recall_at_k(actual: List, predicted: List, k: int = 12) -> float:
    """
    计算召回率@k
    
    Args:
        actual: 真实标签列表
        predicted: 预测标签列表
        k: 截断位置
        
    Returns:
        召回率@k
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    
    if not actual:
        return 0.0
    
    hits = len(set(actual) & set(predicted))
    return hits / len(actual)


def precision_at_k(actual: List, predicted: List, k: int = 12) -> float:
    """
    计算精确率@k
    
    Args:
        actual: 真实标签列表
        predicted: 预测标签列表
        k: 截断位置
        
    Returns:
        精确率@k
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    
    if not predicted:
        return 0.0
    
    hits = len(set(actual) & set(predicted))
    return hits / len(predicted)
