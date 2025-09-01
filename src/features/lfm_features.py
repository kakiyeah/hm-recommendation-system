"""
LightFM特征生成模块

使用LightFM矩阵分解生成用户和商品的低维表示
"""

import os
import pickle
import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy import sparse
from logzero import logger
from typing import Tuple


class LightFMFeatureGenerator:
    """LightFM特征生成器"""
    
    def __init__(self, data_dir: str):
        """
        初始化特征生成器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.lfm_dir = os.path.join(data_dir, "lfm")
        os.makedirs(self.lfm_dir, exist_ok=True)
        
        # LightFM参数
        self.lightfm_params = {
            'learning_schedule': 'adadelta',
            'loss': 'bpr',
            'learning_rate': 0.005,
            'random_state': 42,
        }
        self.epochs = 100
    
    def create_user_item_matrix(self, week: int, dim: int) -> None:
        """
        创建用户-商品矩阵并训练LightFM模型
        
        Args:
            week: 时间窗口
            dim: 嵌入维度
        """
        path_prefix = os.path.join(self.lfm_dir, f"lfm_i_i_week{week}_dim{dim}")
        logger.info(f"生成LightFM特征: {path_prefix}")
        
        # 读取数据
        transactions = pd.read_pickle(os.path.join(self.processed_dir, "transactions_train.pkl"))
        users = pd.read_pickle(os.path.join(self.processed_dir, "users.pkl"))
        items = pd.read_pickle(os.path.join(self.processed_dir, "items.pkl"))
        
        n_user = len(users)
        n_item = len(items)
        
        # 创建用户-商品矩阵
        tr_data = transactions.query(f"@week <= week")[['user', 'item']].drop_duplicates(ignore_index=True)
        user_item_matrix = sparse.lil_matrix((n_user, n_item))
        user_item_matrix[tr_data['user'], tr_data['item']] = 1
        
        # 训练LightFM模型
        lightfm_params = self.lightfm_params.copy()
        lightfm_params['no_components'] = dim
        
        model = LightFM(**lightfm_params)
        model.fit(user_item_matrix, epochs=self.epochs, num_threads=4, verbose=True)
        
        # 保存模型
        save_path = f"{path_prefix}_model.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"LightFM模型已保存: {save_path}")
    
    def generate_embeddings(self, model_type: str, week: int, dim: int) -> pd.DataFrame:
        """
        生成用户嵌入特征
        
        Args:
            model_type: 模型类型
            week: 时间窗口
            dim: 嵌入维度
            
        Returns:
            用户嵌入特征DataFrame
        """
        # 加载模型
        model_path = os.path.join(self.lfm_dir, f"lfm_{model_type}_week{week}_dim{dim}_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 获取用户表示
        biases, embeddings = model.get_user_representations(None)
        n_user = len(biases)
        
        # 合并嵌入和偏置
        user_features = np.hstack([embeddings, biases.reshape(n_user, 1)])
        user_embeddings = pd.DataFrame(
            user_features, 
            columns=[f"user_rep_{i}" for i in range(dim + 1)]
        )
        user_embeddings = pd.concat([
            pd.DataFrame({'user': range(n_user)}), 
            user_embeddings
        ], axis=1)
        
        return user_embeddings
    
    def generate_all_features(self, dim: int = 16) -> None:
        """
        为所有时间窗口生成LightFM特征
        
        Args:
            dim: 嵌入维度
        """
        logger.info("开始生成LightFM特征...")
        
        for week in range(1, 14):
            self.create_user_item_matrix(week, dim)
        
        logger.info("LightFM特征生成完成！")


def main():
    """主函数"""
    data_dir = "data"
    generator = LightFMFeatureGenerator(data_dir)
    generator.generate_all_features()


if __name__ == "__main__":
    main()
