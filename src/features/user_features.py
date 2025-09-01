"""
用户特征生成模块

生成基于用户历史行为的聚合特征
"""

import os
import pandas as pd
import vaex
from logzero import logger
from typing import List


class UserFeatureGenerator:
    """用户特征生成器"""
    
    def __init__(self, data_dir: str):
        """
        初始化特征生成器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.user_features_dir = os.path.join(data_dir, "user_features")
        os.makedirs(self.user_features_dir, exist_ok=True)
    
    def create_user_ohe_agg(self, week: int) -> None:
        """
        对各个item属性特征做onehot编码，并入交易表，然后groupby每个user，
        并且在交易样本中agg平均
        
        Args:
            week: 时间窗口
        """
        # 读取数据
        transactions = pd.read_pickle(os.path.join(self.processed_dir, 'transactions_train.pkl'))[
            ['user', 'item', 'week']
        ]
        users = pd.read_pickle(os.path.join(self.processed_dir, 'users.pkl'))
        items = pd.read_pickle(os.path.join(self.processed_dir, 'items.pkl'))
        
        # 筛选时间窗口内的交易
        tr = vaex.from_pandas(transactions.query(f"week >= @week")[['user', 'item']])
        
        # 获取需要编码的列
        target_columns = [c for c in items.columns if c.endswith('_idx')]
        
        for col in target_columns:
            logger.info(f"处理特征: {col}, week: {week}")
            
            # 加入onehot编码
            tmp = tr.join(
                vaex.from_pandas(pd.get_dummies(items[['item', col]], columns=[col])), 
                on='item'
            )
            tmp = tmp.drop(columns='item')
            
            # 按用户聚合
            tmp = tmp.groupby('user').agg(['mean'])
            
            # 合并到用户表
            users_processed = vaex.from_pandas(users[['user']]).join(
                tmp, on='user', how='left'
            ).to_pandas_df()
            
            # 重命名列
            users_processed = users_processed.rename(columns={
                c: f'user_ohe_agg_{c}' for c in users_processed.columns if c != 'user'
            })
            
            # 排序
            users_processed = users_processed.sort_values(by='user').reset_index(drop=True)
            
            # 保存
            save_path = os.path.join(self.user_features_dir, f'user_ohe_agg_week{week}_{col}.pkl')
            users_processed.to_pickle(save_path)
            logger.info(f"特征已保存: {save_path}")
    
    def generate_all_features(self) -> None:
        """为所有时间窗口生成用户特征"""
        logger.info("开始生成用户特征...")
        
        for week in range(14):
            self.create_user_ohe_agg(week)
        
        logger.info("用户特征生成完成！")


def main():
    """主函数"""
    data_dir = "data"
    generator = UserFeatureGenerator(data_dir)
    generator.generate_all_features()


if __name__ == "__main__":
    main()
