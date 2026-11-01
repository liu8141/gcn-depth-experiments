# GCN Depth Comparison Experiments

TensorFlow-based graph neural network experiments on Cora, PubMed, and CiteSeer, including linear separability metrics.

## Files

| File | Purpose |
| --- | --- |
| `cora_depth_experiment.py` | Cora experiments |
| `citation_depth_experiment (1).py` | PubMed / CiteSeer experiments |
| `Linear_Separability_numpy_mul.py` | Linear separability metrics |
| `run_depths (1).py` | Batch Cora experiments across depths and random seeds |

## Installation

Create a Python virtual environment in the project directory and install dependencies (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependencies are inferred from source imports. Versions have not yet been pinned or validated through a complete training run. Use compatible Python and TensorFlow versions.

## Running Experiments

Run commands from the project root. Datasets are downloaded on the first run.

### Cora

```powershell
$env:GCN_LAYERS = "4"
$env:GCN_SEED = "42"
.\.venv\Scripts\python.exe cora_depth_experiment.py
```

Defaults are 16 layers, random seed 42, and 300 epochs. Results are saved to `results_pkls/`.

Batch execution:

```powershell
.\.venv\Scripts\python.exe "run_depths (1).py" --layers 2 4 6 8 --seeds 42
```

Logs, metrics, and the summary table are saved to `depth_runs/`. The batch runner searches for Cora-specific result filenames and does not support the PubMed / CiteSeer script.

### PubMed / CiteSeer

```powershell
$env:GCN_DATASET = "pubmed" # Change to citeseer as needed.
$env:GCN_LAYERS = "4"
$env:GCN_SEED = "42"
$env:GCN_EPOCHS = "300"
$env:GCN_RESULT_DIR = "results_pkls/pubmed_layers4_seed42"
.\.venv\Scripts\python.exe "citation_depth_experiment (1).py"
```

Datasets are cached in `data/` by default; override this location with `GCN_DATA_DIR`. Results are written to `GCN_RESULT_DIR/metrics.pkl`. Use a separate result directory for each experiment to avoid overwriting previous results.

## Notes

- Data is randomly split into 70% training, 10% validation, and 20% test sets using the configured seed.
- Checkpoints are selected by validation accuracy before evaluating the test set.
- Downloaded datasets, training outputs, and local environments are excluded from Git.
- This repository does not currently include an open-source license.
