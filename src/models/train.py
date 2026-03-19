"""
模型训练脚本
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

import wandb

from src.models.gnn import CGCNN, MaterialsTransformer
from src.data.data_loader import MaterialsDataset


def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            out = model(batch)
            
            predictions.extend(out.cpu().numpy().flatten())
            targets.extend(batch.y.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return mae, r2, predictions, targets


def main(config):
    """主训练函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化 wandb
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'ml-materials'),
            name=config.get('experiment_name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config=config
        )
    
    # 加载数据
    print("加载数据集...")
    dataset = MaterialsDataset(
        data_path=config['data_path'],
        target_property=config.get('target_property', 'formation_energy_per_atom')
    )
    
    # 划分训练/验证/测试集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # 创建模型
    print("创建模型...")
    model_type = config.get('model_type', 'cgcnn')
    
    if model_type == 'cgcnn':
        model = CGCNN(
            node_dim=config.get('node_dim', 92),
            edge_dim=config.get('edge_dim', 41),
            hidden_dim=config.get('hidden_dim', 64),
            num_conv_layers=config.get('num_conv_layers', 3),
            num_fc_layers=config.get('num_fc_layers', 2),
            num_targets=config.get('num_targets', 1)
        )
    elif model_type == 'transformer':
        model = MaterialsTransformer(
            node_dim=config.get('node_dim', 92),
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_encoder_layers=config.get('num_encoder_layers', 6),
            num_targets=config.get('num_targets', 1)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # 训练循环
    best_val_mae = float('inf')
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config.get('num_epochs', 100)):
        print(f"\nEpoch {epoch + 1}/{config.get('num_epochs', 100)}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_mae, val_r2, _, _ = evaluate(model, val_loader, device)
        
        # 学习率调整
        scheduler.step(val_mae)
        
        # 记录日志
        print(f"Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")
        
        if config.get('use_wandb', False):
            wandb.log({
                'train_loss': train_loss,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'config': config
            }, output_dir / 'best_model.pt')
            print(f"保存最佳模型，Val MAE: {val_mae:.4f}")
    
    # 测试
    print("\n测试最佳模型...")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_mae, test_r2, predictions, targets = evaluate(model, test_loader, device)
    print(f"Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")
    
    if config.get('use_wandb', False):
        wandb.log({'test_mae': test_mae, 'test_r2': test_r2})
        wandb.finish()
    
    # 保存预测结果
    results = {
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    
    import json
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n训练完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练材料性质预测模型")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
