# H&M Personalized Fashion Recommendations

## 项目概述

基于Kaggle H&M个性化时尚推荐竞赛的推荐系统项目。

## 项目特点

- **多策略候选生成**: 结合重购策略、热门商品策略、类别热门策略等
- **丰富的特征工程**: 用户特征、商品特征、交互特征、时间特征等
- **先进的推荐算法**: LightFM矩阵分解 + CatBoost排序模型
- **完整的推荐流程**: 从数据处理到模型训练再到预测推理

## 技术栈

- **数据处理**: Pandas, NumPy, Vaex
- **推荐算法**: LightFM (矩阵分解)
- **机器学习**: CatBoost, LightGBM
- **特征工程**: 标签编码、One-hot编码、聚合特征
- **评估指标**: MAP@12 (Mean Average Precision at 12)

## START

### 1. 环境准备

```bash

python -m venv venv
source venv/bin/activate  

pip install -r requirements.txt
```

### 2. 数据准备

将H&M竞赛的原始数据文件放置在 `data/raw/` 目录下：
- `articles.csv`
- `customers.csv` 
- `transactions_train.csv`

### 3. 运行项目

```bash
# 数据预处理
python src/data/preprocessing.py

# 特征工程
python src/features/lfm_features.py
python src/features/user_features.py

# 模型训练
python src/models/training.py
```

## 核心算法

### 1. 候选生成策略

- **重购策略 (Repurchase)**: 基于用户历史购买记录
- **热门商品策略 (Popular)**: 基于全局热门商品
- **类别热门策略 (Category Popular)**: 基于特定类别的热门商品
- **Item2Item策略**: 基于相似商品的推荐

### 2. 特征工程

- **用户特征**: 年龄、会员状态、购买行为等
- **商品特征**: 类别、颜色、部门等属性
- **交互特征**: 用户-商品交互历史
- **时间特征**: 购买时间、商品新鲜度等
- **嵌入特征**: LightFM生成的用户和商品嵌入

### 3. 模型架构

1. **LightFM矩阵分解**: 生成用户和商品的低维表示
2. **CatBoost排序模型**: 对候选商品进行排序
3. **多策略融合**: 结合多种推荐策略的结果

## 性能表现

- **MAP@12**: 0.0234 (在验证集上的表现)
- **模型**: CatBoost + LightFM
- **特征数量**: 100+ 个特征
- **候选商品数**: 平均每个用户60个候选商品



