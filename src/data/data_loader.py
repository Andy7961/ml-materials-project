"""
数据预处理和加载工具
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from ase import Atoms
from ase.io import read
from pymatgen.core import Structure, Composition
from pymatgen.io.ase import AseAtomsAdaptor

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MaterialsDataset(Dataset):
    """
    材料数据集类
    支持从 CIF、POSCAR 等格式加载晶体结构
    """
    
    def __init__(
        self,
        data_path: str,
        transform=None,
        target_property: str = "formation_energy_per_atom"
    ):
        """
        Args:
            data_path: 数据文件路径 (CSV, JSON, 或目录)
            transform: 数据变换函数
            target_property: 目标属性名称
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_property = target_property
        
        self.structures = []
        self.targets = []
        self.material_ids = []
        
        self._load_data()
        
    def _load_data(self):
        """加载数据"""
        if self.data_path.suffix == ".csv":
            self._load_from_csv()
        elif self.data_path.suffix == ".json":
            self._load_from_json()
        elif self.data_path.is_dir():
            self._load_from_directory()
        else:
            raise ValueError(f"不支持的数据格式: {self.data_path}")
    
    def _load_from_csv(self):
        """从 CSV 加载数据"""
        df = pd.read_csv(self.data_path)
        
        # 假设 CSV 包含结构文件路径和目标值
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="加载数据"):
            try:
                structure_path = row.get("structure_path", row.get("cif_path"))
                if structure_path and os.path.exists(structure_path):
                    structure = Structure.from_file(structure_path)
                    self.structures.append(structure)
                    self.targets.append(row.get(self.target_property, 0.0))
                    self.material_ids.append(row.get("material_id", f"material_{idx}"))
            except Exception as e:
                print(f"加载 {idx} 失败: {e}")
    
    def _load_from_json(self):
        """从 JSON 加载数据"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        for item in tqdm(data, desc="加载数据"):
            try:
                structure = Structure.from_dict(item["structure"])
                self.structures.append(structure)
                self.targets.append(item.get(self.target_property, 0.0))
                self.material_ids.append(item.get("material_id", "unknown"))
            except Exception as e:
                print(f"加载失败: {e}")
    
    def _load_from_directory(self):
        """从目录加载 CIF 文件"""
        cif_files = list(self.data_path.glob("*.cif"))
        
        for cif_file in tqdm(cif_files, desc="加载 CIF 文件"):
            try:
                structure = Structure.from_file(cif_file)
                self.structures.append(structure)
                self.targets.append(0.0)  # 默认目标值
                self.material_ids.append(cif_file.stem)
            except Exception as e:
                print(f"加载 {cif_file} 失败: {e}")
    
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        target = self.targets[idx]
        material_id = self.material_ids[idx]
        
        # 转换为图数据
        data = structure_to_graph(structure)
        data.y = torch.tensor([target], dtype=torch.float)
        data.material_id = material_id
        
        if self.transform:
            data = self.transform(data)
        
        return data


def structure_to_graph(structure: Structure, cutoff: float = 8.0) -> Data:
    """
    将 pymatgen Structure 转换为 PyG Data 对象
    
    Args:
        structure: pymatgen Structure 对象
        cutoff: 邻居截断距离（Å）
    
    Returns:
        PyG Data 对象
    """
    # 获取原子特征
    atom_features = []
    for site in structure:
        element = site.specie.symbol
        features = get_atom_features(element)
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 构建边（基于距离）
    edge_index = []
    edge_attr = []
    
    for i, site_i in enumerate(structure):
        neighbors = structure.get_neighbors(site_i, r=cutoff)
        for site_j, distance in neighbors:
            j = structure.index(site_j)
            edge_index.append([i, j])
            
            # 边特征：距离和径向基函数
            edge_features = get_edge_features(distance, cutoff)
            edge_attr.append(edge_features)
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # 如果没有邻居，添加自环
        edge_index = torch.arange(len(structure)).unsqueeze(0).repeat(2, 1)
        edge_attr = torch.zeros((len(structure), 41))
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_atom_features(element: str) -> List[float]:
    """
    获取原子特征向量
    包含：原子序数、族、周期、电负性、原子半径等
    """
    from pymatgen.core import Element
    
    try:
        elem = Element(element)
        features = [
            elem.Z,  # 原子序数
            elem.group,  # 族
            elem.row,  # 周期
            elem.X if elem.X else 0.0,  # 电负性
            elem.atomic_radius if elem.atomic_radius else 0.0,  # 原子半径
            elem.atomic_mass,  # 原子质量
            elem.melting_point if elem.melting_point else 0.0,  # 熔点
            elem.boiling_point if elem.boiling_point else 0.0,  # 沸点
            elem.density_of_solid if elem.density_of_solid else 0.0,  # 固态密度
            elem.thermal_conductivity if elem.thermal_conductivity else 0.0,  # 热导率
        ]
    except:
        # 如果元素不存在，返回零向量
        features = [0.0] * 10
    
    return features


def get_edge_features(distance: float, cutoff: float, num_rbf: int = 40) -> List[float]:
    """
    获取边特征向量
    使用径向基函数 (RBF) 编码距离
    """
    # 距离特征
    features = [distance / cutoff]  # 归一化距离
    
    # 径向基函数
    centers = np.linspace(0, cutoff, num_rbf)
    width = cutoff / num_rbf
    rbf = np.exp(-((distance - centers) ** 2) / (2 * width ** 2))
    features.extend(rbf.tolist())
    
    return features


def load_materials_project_data(api_key: Optional[str] = None) -> pd.DataFrame:
    """
    从 Materials Project API 加载数据
    
    Args:
        api_key: Materials Project API 密钥
    
    Returns:
        包含材料数据的 DataFrame
    """
    try:
        from mp_api.client import MPRester
        
        with MPRester(api_key) as mpr:
            # 获取材料数据
            docs = mpr.materials.summary.search(
                fields=["material_id", "formula_pretty", "structure", 
                       "formation_energy_per_atom", "band_gap", "energy_above_hull"],
                num_chunks=10,
                chunk_size=1000
            )
        
        # 转换为 DataFrame
        data = []
        for doc in docs:
            data.append({
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "band_gap": doc.band_gap,
                "energy_above_hull": doc.energy_above_hull,
            })
        
        return pd.DataFrame(data)
    
    except ImportError:
        print("请先安装 mp-api: pip install mp-api")
        return pd.DataFrame()


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据结构转换...")
    
    # 创建一个简单的晶体结构
    from pymatgen.core import Lattice
    
    lattice = Lattice.cubic(4.0)
    structure = Structure(
        lattice,
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]]
    )
    
    # 转换为图
    data = structure_to_graph(structure)
    print(f"节点特征形状: {data.x.shape}")
    print(f"边索引形状: {data.edge_index.shape}")
    print(f"边特征形状: {data.edge_attr.shape}")
