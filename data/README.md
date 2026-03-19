# 数据目录

此目录用于存放数据文件。

## 子目录说明

- `raw/` - 原始数据文件
- `processed/` - 处理后的数据文件
- `external/` - 外部数据源

## 数据来源

1. **Materials Project**: https://materialsproject.org/
2. **AFLOW**: http://aflowlib.org/
3. **OQMD**: http://oqmd.org/

## 数据格式

### CSV 格式
```csv
material_id,formula,structure_path,formation_energy_per_atom,band_gap
mp-1234,SiO2,data/raw/mp-1234.cif,-3.5,5.2
```

### JSON 格式
```json
[
  {
    "material_id": "mp-1234",
    "formula": "SiO2",
    "structure": {...},
    "formation_energy_per_atom": -3.5,
    "band_gap": 5.2
  }
]
```
