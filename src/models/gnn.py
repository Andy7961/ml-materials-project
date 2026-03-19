"""
晶体图神经网络 (Crystal Graph Neural Network) 模型
用于预测材料性质
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops


class CGCNNConv(MessagePassing):
    """
    CGCNN 图卷积层
    参考: Crystal Graph Convolutional Neural Networks (Xie & Grossman, 2018)
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(CGCNNConv, self).__init__(aggr='add')
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # 边特征转换
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        # 节点更新
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: 节点特征 [N, node_dim]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """
        消息函数
        Args:
            x_i: 源节点特征 [E, node_dim]
            x_j: 目标节点特征 [E, node_dim]
            edge_attr: 边特征 [E, edge_dim]
        """
        # 边特征转换
        edge_features = self.edge_mlp(edge_attr)
        
        # 拼接节点和边特征
        message = torch.cat([x_i, x_j, edge_features], dim=-1)
        return self.node_mlp(message)
    
    def update(self, aggr_out, x):
        """更新节点特征"""
        return x + aggr_out  # 残差连接


class CGCNN(nn.Module):
    """
    完整的 CGCNN 模型
    """
    
    def __init__(
        self,
        node_dim=92,  # 原子特征维度
        edge_dim=41,  # 边特征维度
        hidden_dim=64,
        num_conv_layers=3,
        num_fc_layers=2,
        num_targets=1
    ):
        super(CGCNN, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # 嵌入层
        self.embedding = nn.Linear(node_dim, hidden_dim)
        
        # 图卷积层
        self.conv_layers = nn.ModuleList([
            CGCNNConv(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_conv_layers)
        ])
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_conv_layers)
        ])
        
        # 全连接层
        fc_dims = [hidden_dim * 2] + [hidden_dim] * num_fc_layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_dims) - 1):
            self.fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i+1]))
        
        # 输出层
        self.output_layer = nn.Linear(fc_dims[-1], num_targets)
        
    def forward(self, data):
        """
        前向传播
        Args:
            data: PyG Data 对象，包含 x, edge_index, edge_attr, batch
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 节点嵌入
        x = self.embedding(x)
        
        # 图卷积
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.softplus(x)
        
        # 全局池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_mean_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        # 全连接层
        for fc in self.fc_layers:
            x = F.softplus(fc(x))
        
        # 输出
        out = self.output_layer(x)
        return out


class MaterialsTransformer(nn.Module):
    """
    基于 Transformer 的材料性质预测模型
    将晶体结构视为序列进行处理
    """
    
    def __init__(
        self,
        node_dim=92,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        num_targets=1,
        dropout=0.1
    ):
        super(MaterialsTransformer, self).__init__()
        
        # 输入嵌入
        self.embedding = nn.Linear(node_dim, d_model)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_targets)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入特征 [batch_size, seq_len, node_dim]
            mask: 注意力掩码 [batch_size, seq_len]
        """
        # 嵌入
        x = self.embedding(x)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # 全局平均池化
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # 输出
        out = self.output_head(x)
        return out


def test_model():
    """测试模型"""
    from torch_geometric.data import Data, Batch
    
    # 创建测试数据
    x = torch.randn(10, 92)  # 10个原子，92维特征
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edge_attr = torch.randn(5, 41)  # 5条边，41维特征
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data, data])
    
    # 测试 CGCNN
    model = CGCNN(node_dim=92, edge_dim=41, num_targets=1)
    output = model(batch)
    print(f"CGCNN 输出形状: {output.shape}")
    
    # 测试 Transformer
    transformer = MaterialsTransformer(node_dim=92, num_targets=1)
    x_seq = torch.randn(2, 10, 92)  # batch_size=2, seq_len=10
    output = transformer(x_seq)
    print(f"Transformer 输出形状: {output.shape}")


if __name__ == "__main__":
    test_model()
