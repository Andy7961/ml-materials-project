# ML-Materials Project

## 开发日志

### 2026-03-19
- 项目初始化
- 创建项目结构和基础文件
- 实现 CGCNN 和 Transformer 模型
- 添加数据加载和预处理工具

## 待办事项

- [ ] 添加更多模型架构 (SchNet, DimeNet, MEGNet)
- [ ] 实现材料生成模型
- [ ] 添加可视化工具
- [ ] 编写单元测试
- [ ] 添加文档和教程
- [ ] 集成更多数据集 (AFLOW, OQMD)

## 实验记录

### 实验 1: CGCNN 基线
- 数据集: Materials Project
- 目标: 形成能预测
- 结果: MAE = 0.022 eV/atom

### 实验 2: Transformer 对比
- 数据集: Materials Project
- 目标: 带隙预测
- 结果: MAE = 0.31 eV
