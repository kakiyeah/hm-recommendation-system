"""
数据预处理模块

将原始数据转换为模型训练所需的格式，包括：
- 标签编码
- 数据类型转换
- 缺失值处理
- 时间特征生成
"""

import os
import pandas as pd
from logzero import logger
from typing import Dict, Any
import pickle

# 数据格式定义
ARTICLES_ORIGINAL = {
    'article_id': 'object',
    'product_code': 'int64',
    'prod_name': 'object',
    'product_type_no': 'int64',
    'product_type_name': 'object',
    'product_group_name': 'object',
    'graphical_appearance_no': 'int64',
    'graphical_appearance_name': 'object',
    'colour_group_code': 'int64',
    'colour_group_name': 'object',
    'perceived_colour_value_id': 'int64',
    'perceived_colour_value_name': 'object',
    'perceived_colour_master_id': 'int64',
    'perceived_colour_master_name': 'object',
    'department_no': 'int64',
    'department_name': 'object',
    'index_code': 'object',
    'index_name': 'object',
    'index_group_no': 'int64',
    'index_group_name': 'object',
    'section_no': 'int64',
    'section_name': 'object',
    'garment_group_no': 'int64',
    'garment_group_name': 'object',
    'detail_desc': 'object',
}

CUSTOMERS_ORIGINAL = {
    'customer_id': 'object',
    'FN': 'float64',
    'Active': 'float64',
    'club_member_status': 'object',
    'fashion_news_frequency': 'object',
    'age': 'float64',
    'postal_code': 'object',
}

TRANSACTIONS_ORIGINAL = {
    'customer_id': 'object',
    'article_id': 'object',
    'price': 'float64',
    'sales_channel_id': 'int64',
}


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_dir: str):
        """
        初始化预处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def _count_encoding_dict(self, df: pd.DataFrame, col_name: str) -> Dict[Any, int]:
        """
        根据列的value count排序，生成标签编码字典
        
        Args:
            df: 数据框
            col_name: 列名
            
        Returns:
            标签编码字典
        """
        v = df.groupby(col_name).size().reset_index(name='size').sort_values(
            by='size', ascending=False)[col_name].tolist()
        return {x: i for i, x in enumerate(v)}
    
    def _dict_to_dataframe(self, mapping: Dict[Any, int]) -> pd.DataFrame:
        """字典转DataFrame"""
        return pd.DataFrame(mapping.items(), columns=['val', 'idx'])
    
    def _add_idx_column(self, df: pd.DataFrame, col_name_from: str, 
                       col_name_to: str, mapping: Dict[Any, int]) -> None:
        """添加标签编码列"""
        df[col_name_to] = df[col_name_from].apply(lambda x: mapping[x]).astype('int64')
    
    def process_data(self) -> None:
        """执行完整的数据预处理流程"""
        logger.info("开始数据预处理...")
        
        # 读取原始数据
        logger.info("读取原始数据...")
        articles = pd.read_csv(
            os.path.join(self.data_dir, 'raw', 'articles.csv'), 
            dtype=ARTICLES_ORIGINAL
        )
        customers = pd.read_csv(
            os.path.join(self.data_dir, 'raw', 'customers.csv'), 
            dtype=CUSTOMERS_ORIGINAL
        )
        transactions = pd.read_csv(
            os.path.join(self.data_dir, 'raw', 'transactions_train.csv'),
            dtype=TRANSACTIONS_ORIGINAL,
            parse_dates=['t_dat']
        )
        
        # 生成ID映射
        logger.info("生成ID映射...")
        customer_ids = customers.customer_id.values
        mp_customer_id = {x: i for i, x in enumerate(customer_ids)}
        
        article_ids = articles.article_id.values
        mp_article_id = {x: i for i, x in enumerate(article_ids)}
        
        # 保存映射
        self._dict_to_dataframe(mp_customer_id).to_pickle(
            os.path.join(self.processed_dir, 'mp_customer_id.pkl')
        )
        self._dict_to_dataframe(mp_article_id).to_pickle(
            os.path.join(self.processed_dir, 'mp_article_id.pkl')
        )
        
        # 处理customers数据
        logger.info("处理customers数据...")
        self._add_idx_column(customers, 'customer_id', 'user', mp_customer_id)
        
        # 处理缺失值
        customers['FN'] = customers['FN'].fillna(0).astype('int64')
        customers['Active'] = customers['Active'].fillna(0).astype('int64')
        customers['club_member_status'] = customers['club_member_status'].fillna('NULL')
        customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NULL')
        
        # 标签编码
        for col_name in ['club_member_status', 'fashion_news_frequency']:
            mp = self._count_encoding_dict(customers, col_name)
            self._add_idx_column(customers, col_name, f'{col_name}_idx', mp)
        
        customers.to_pickle(os.path.join(self.processed_dir, 'users.pkl'))
        
        # 处理articles数据
        logger.info("处理articles数据...")
        self._add_idx_column(articles, 'article_id', 'item', mp_article_id)
        
        # 标签编码
        count_encoding_columns = [
            'product_type_no', 'product_group_name', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id',
            'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no',
        ]
        
        for col_name in count_encoding_columns:
            mp = self._count_encoding_dict(articles, col_name)
            self._add_idx_column(articles, col_name, f'{col_name}_idx', mp)
        
        articles.to_pickle(os.path.join(self.processed_dir, 'items.pkl'))
        
        # 处理transactions数据
        logger.info("处理transactions数据...")
        self._add_idx_column(transactions, 'customer_id', 'user', mp_customer_id)
        self._add_idx_column(transactions, 'article_id', 'item', mp_article_id)
        
        # 调整sales_channel_id
        transactions['sales_channel_id'] = transactions['sales_channel_id'] - 1
        
        # 生成时间特征
        transactions['week'] = (transactions['t_dat'].max() - transactions['t_dat']).dt.days // 7
        transactions['day'] = (transactions['t_dat'].max() - transactions['t_dat']).dt.days
        
        transactions.to_pickle(os.path.join(self.processed_dir, 'transactions_train.pkl'))
        
        logger.info("数据预处理完成！")


def main():
    """主函数"""
    data_dir = "data"
    preprocessor = DataPreprocessor(data_dir)
    preprocessor.process_data()


if __name__ == "__main__":
    main()
