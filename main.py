#!/usr/bin/env python3
"""
H&M推荐系统主运行脚本

运行完整的推荐系统流程：
1. 数据预处理
2. 特征工程
3. 模型训练
4. 预测推理
"""

import os
import sys
import argparse
from logzero import logger

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.preprocessing import DataPreprocessor
from features.lfm_features import LightFMFeatureGenerator
from features.user_features import UserFeatureGenerator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='H&M推荐系统')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='数据目录路径')
    parser.add_argument('--step', type=str, choices=['preprocess', 'features', 'train', 'all'],
                       default='all', help='运行步骤')
    
    args = parser.parse_args()
    
    logger.info("开始运行H&M推荐系统...")
    
    if args.step in ['preprocess', 'all']:
        logger.info("步骤1: 数据预处理")
        preprocessor = DataPreprocessor(args.data_dir)
        preprocessor.process_data()
    
    if args.step in ['features', 'all']:
        logger.info("步骤2: 特征工程")
        
        # LightFM特征
        logger.info("生成LightFM特征...")
        lfm_generator = LightFMFeatureGenerator(args.data_dir)
        lfm_generator.generate_all_features()
        
        # 用户特征
        logger.info("生成用户特征...")
        user_generator = UserFeatureGenerator(args.data_dir)
        user_generator.generate_all_features()
    
    if args.step in ['train', 'all']:
        logger.info("步骤3: 模型训练")
        # TODO: 实现模型训练
        logger.info("模型训练功能待实现")
    
    logger.info("H&M推荐系统运行完成！")


if __name__ == "__main__":
    main()
