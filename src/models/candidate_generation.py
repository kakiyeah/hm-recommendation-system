"""
候选生成模块

生成推荐候选商品，包括多种策略：
- 重购策略
- 热门商品策略
- 类别热门策略
- Item2Item策略
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from logzero import logger


class CandidateGenerator:
    """候选生成器"""
    
    def __init__(self, transactions: pd.DataFrame, items: pd.DataFrame):
        """
        初始化候选生成器
        
        Args:
            transactions: 交易数据
            items: 商品数据
        """
        self.transactions = transactions
        self.items = items
    
    def create_candidates_repurchase(
        self, 
        strategy: str,
        target_users: np.ndarray,
        week_start: int,
        max_items_per_user: int = 1234567890
    ) -> pd.DataFrame:
        """
        创建重购候选
        
        Args:
            strategy: 策略名称
            target_users: 目标用户
            week_start: 开始周数
            max_items_per_user: 每个用户最大商品数
            
        Returns:
            候选商品DataFrame
        """
        # 筛选交易数据
        tr = self.transactions.query(
            f"user in @target_users and @week_start <= week"
        )[['user', 'item', 'week', 'day']].drop_duplicates(ignore_index=True)
        
        # 计算各种排名
        gr_day = tr.groupby(['user', 'item'])['day'].min().reset_index(name='day')
        gr_week = tr.groupby(['user', 'item'])['week'].min().reset_index(name='week')
        gr_volume = tr.groupby(['user', 'item']).size().reset_index(name='volume')
        
        gr_day['day_rank'] = gr_day.groupby('user')['day'].rank()
        gr_week['week_rank'] = gr_week.groupby('user')['week'].rank()
        gr_volume['volume_rank'] = gr_volume.groupby('user')['volume'].rank(ascending=False)
        
        # 合并排名
        candidates = gr_day.merge(gr_week, on=['user', 'item']).merge(gr_volume, on=['user', 'item'])
        
        # 计算综合排名
        candidates['rank_meta'] = 10**9 * candidates['day_rank'] + candidates['volume_rank']
        candidates['rank_meta'] = candidates.groupby('user')['rank_meta'].rank(method='min')
        
        # 筛选top商品
        candidates = candidates.query(f"rank_meta <= @max_items_per_user").reset_index(drop=True)
        
        # 整理列名
        candidates = candidates[['user', 'item', 'week_rank', 'volume_rank', 'rank_meta']].rename(
            columns={'week_rank': f'{strategy}_week_rank', 'volume_rank': f'{strategy}_volume_rank'}
        )
        candidates['strategy'] = strategy
        
        return candidates.drop_duplicates(ignore_index=True)
    
    def create_candidates_popular(
        self,
        target_users: np.ndarray,
        week_start: int,
        num_weeks: int,
        num_items: int
    ) -> pd.DataFrame:
        """
        创建热门商品候选
        
        Args:
            target_users: 目标用户
            week_start: 开始周数
            num_weeks: 周数范围
            num_items: 热门商品数量
            
        Returns:
            候选商品DataFrame
        """
        # 筛选时间窗口内的交易
        tr = self.transactions.query(
            f"@week_start <= week < @week_start + @num_weeks"
        )[['user', 'item']].drop_duplicates(ignore_index=True)
        
        # 获取热门商品
        popular_items = tr['item'].value_counts().index.values[:num_items]
        popular_items_df = pd.DataFrame({
            'item': popular_items,
            'rank': range(num_items),
            'crossjoinkey': 1,
        })
        
        # 创建用户-商品笛卡尔积
        candidates = pd.DataFrame({
            'user': target_users,
            'crossjoinkey': 1,
        })
        
        candidates = candidates.merge(popular_items_df, on='crossjoinkey').drop('crossjoinkey', axis=1)
        candidates = candidates.rename(columns={'rank': 'pop_rank'})
        candidates['strategy'] = 'pop'
        
        return candidates.drop_duplicates(ignore_index=True)
    
    def create_candidates_category_popular(
        self,
        base_candidates: pd.DataFrame,
        week_start: int,
        num_weeks: int,
        num_items_per_category: int,
        category: str
    ) -> pd.DataFrame:
        """
        创建类别热门候选
        
        Args:
            base_candidates: 基础候选
            week_start: 开始周数
            num_weeks: 周数范围
            num_items_per_category: 每个类别商品数
            category: 类别列名
            
        Returns:
            候选商品DataFrame
        """
        # 计算类别内热门商品
        tr = self.transactions.query(
            f"@week_start <= week < @week_start + @num_weeks"
        )[['user', 'item']].drop_duplicates()
        
        tr = tr.groupby('item').size().reset_index(name='volume')
        tr = tr.merge(self.items[['item', category]], on='item')
        tr['cat_volume_rank'] = tr.groupby(category)['volume'].rank(ascending=False, method='min')
        tr = tr.query(f"cat_volume_rank <= @num_items_per_category").reset_index(drop=True)
        tr = tr[['item', category, 'cat_volume_rank']].reset_index(drop=True)
        
        # 合并到基础候选
        candidates = base_candidates[['user', 'item']].merge(
            self.items[['item', category]], on='item'
        )
        candidates = candidates.groupby(['user', category]).size().reset_index(name='cat_volume')
        candidates = candidates.merge(tr, on=category).drop(category, axis=1)
        candidates['strategy'] = 'cat_pop'
        
        return candidates
    
    def drop_common_user_item(
        self, 
        candidates_target: pd.DataFrame, 
        candidates_reference: pd.DataFrame
    ) -> pd.DataFrame:
        """
        从目标候选中去掉参考候选中出现的user-item对
        
        Args:
            candidates_target: 目标候选
            candidates_reference: 参考候选
            
        Returns:
            过滤后的候选
        """
        tmp = candidates_reference[['user', 'item']].reset_index(drop=True)
        tmp['flag'] = 1
        candidates = candidates_target.merge(tmp, on=['user', 'item'], how='left')
        return candidates.query("flag != 1").reset_index(drop=True).drop('flag', axis=1)
    
    def create_candidates(
        self, 
        target_users: np.ndarray, 
        week: int,
        popular_num_items: int = 60,
        popular_weeks: int = 1,
        item2item_num_items: int = 12
    ) -> pd.DataFrame:
        """
        创建综合候选
        
        Args:
            target_users: 目标用户
            week: 时间窗口
            popular_num_items: 热门商品数量
            popular_weeks: 热门商品时间窗口
            item2item_num_items: Item2Item商品数量
            
        Returns:
            综合候选DataFrame
        """
        logger.info(f"创建候选 (week: {week})")
        
        # 重购候选
        candidates_repurchase = self.create_candidates_repurchase(
            'repurchase', target_users, week
        )
        
        # 热门候选
        candidates_popular = self.create_candidates_popular(
            target_users, week, popular_weeks, popular_num_items
        )
        
        # Item2Item候选
        candidates_item2item2 = self.create_candidates_repurchase(
            'item2item2', target_users, week, item2item_num_items
        )
        
        # 类别热门候选
        candidates_dept = self.create_candidates_category_popular(
            self.transactions, self.items, candidates_item2item2, week, 1, 6, 'department_no_idx'
        )
        candidates_dept = self.drop_common_user_item(candidates_dept, candidates_repurchase)
        
        # 合并所有候选
        candidates = pd.concat([
            candidates_repurchase,
            candidates_popular,
            candidates_dept,
        ])
        
        # 统计信息
        logger.info(f"候选数量: {len(candidates)}")
        logger.info(f"重复率: {len(candidates) / len(candidates[['user', 'item']].drop_duplicates())}")
        
        volumes = candidates.groupby('strategy').size().reset_index(name='volume').sort_values(
            by='volume', ascending=False
        ).reset_index(drop=True)
        volumes['ratio'] = volumes['volume'] / volumes['volume'].sum()
        logger.info(f"策略分布:\n{volumes}")
        
        # 删除meta列
        meta_columns = [c for c in candidates.columns if c.endswith('_meta')]
        return candidates.drop(meta_columns, axis=1)
