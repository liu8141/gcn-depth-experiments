# GCN 深度对比实验

基于 TensorFlow 的图神经网络实验代码，包含 Cora、PubMed 和 CiteSeer 数据集实验，以及线性可分性指标计算。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `cora_depth_experiment.py` | Cora 实验 |
| `citation_depth_experiment (1).py` | PubMed / CiteSeer 实验 |
| `Linear_Separability_numpy_mul.py` | 线性可分性指标计算 |
| `run_depths (1).py` | 按层数和随机种子批量运行 Cora |

## 安装

在项目目录创建 Python 虚拟环境并安装依赖（以下命令适用于 Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依赖清单根据源码导入整理，尚未锁定经过完整训练验证的版本。Python 和 TensorFlow 版本需相互兼容。

## 运行

请从项目根目录运行。首次运行会联网下载对应数据集。

### Cora

```powershell
$env:GCN_LAYERS = "4"
$env:GCN_SEED = "42"
.\.venv\Scripts\python.exe cora_depth_experiment.py
```

默认层数为 16，随机种子为 42，训练轮数为 300。结果保存至 `results_pkls/`。

批量运行 Cora：

```powershell
.\.venv\Scripts\python.exe "run_depths (1).py" --layers 2 4 6 8 --seeds 42
```

批量运行的日志、指标和汇总表保存在 `depth_runs/`。当前批量脚本按 Cora 结果命名查找文件，不适用于 PubMed / CiteSeer 脚本。

### PubMed / CiteSeer

```powershell
$env:GCN_DATASET = "pubmed" # 可改为 citeseer
$env:GCN_LAYERS = "4"
$env:GCN_SEED = "42"
$env:GCN_EPOCHS = "300"
$env:GCN_RESULT_DIR = "results_pkls/pubmed_layers4_seed42"
.\.venv\Scripts\python.exe "citation_depth_experiment (1).py"
```

数据默认缓存至 `data/`，可通过 `GCN_DATA_DIR` 修改。结果写入 `GCN_RESULT_DIR/metrics.pkl`；每次实验应指定不同目录，避免覆盖。

## 说明

- 数据按随机种子划分为训练集 70%、验证集 10%、测试集 20%。
- 使用验证准确率选择模型检查点，再评估测试集。
- 数据集、训练产物及本地环境不提交到 Git。
- 本仓库尚未添加开源许可证。
