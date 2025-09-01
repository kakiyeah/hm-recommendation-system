# H&M Personalized Fashion Recommendations

## 项目概述

这是一个基于Kaggle H&M个性化时尚推荐竞赛的推荐系统项目。该项目使用多种推荐策略和机器学习模型来预测用户可能购买的时尚商品。

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

## 项目结构

```
hm-recommendation-system/
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── setup.py                 # 项目安装配置
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后的数据
├── src/                     # 源代码
│   ├── __init__.py
│   ├── data/               # 数据处理模块
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/           # 特征工程模块
│   │   ├── __init__.py
│   │   ├── lfm_features.py
│   │   └── user_features.py
│   ├── models/             # 模型模块
│   │   ├── __init__.py
│   │   ├── candidate_generation.py
│   │   ├── feature_engineering.py
│   │   └── training.py
│   └── utils/              # 工具函数
│       ├── __init__.py
│       └── metrics.py
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── configs/                # 配置文件
│   └── config.yaml
├── models/                 # 训练好的模型
├── results/                # 结果文件
└── tests/                  # 测试文件
    └── __init__.py
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/hm-recommendation-system.git
cd hm-recommendation-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
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

## 项目亮点

1. **完整的推荐系统流程**: 从数据处理到模型部署的完整pipeline
2. **多策略融合**: 结合多种推荐策略提高推荐多样性
3. **丰富的特征工程**: 充分利用用户、商品、交互和时间信息
4. **高效的候选生成**: 通过多种策略平衡推荐准确性和覆盖率
5. **可扩展的架构**: 模块化设计便于扩展和维护

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License

## 联系方式

如有问题，请通过以下方式联系：
- Email: your-email@example.com
- GitHub: https://github.com/kakiyeah
