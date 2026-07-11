# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HCNNs and Chaos** is a PhD research project modeling chaotic dynamical systems using:
- **HCNNs** (Historical Consistent Neural Networks) - Vanilla, PTF (Partial Teacher Forcing), LForm (LSTM Formulation), LSpa (Large Sparse)
- **Classic RNNs and LSTMs**

The codebase supports both single-model and ensemble training pipelines implemented in PyTorch with GPU/MPS optimization.

## Repository Structure

```
src/                              # Main unified modeling stack
├── models/
│   ├── classic.py               # RNN_Model, LSTM_Model
│   └── HCNN/
│       ├── hcnn_models.py       # Vanilla, PTF, LForm, LSpa model implementations
│       └── modules.py           # Core HCNN module definitions
├── ensembles/
│   ├── classics.py              # RNN and LSTM ensemble classes
│   └── hcnns.py                 # HCNN ensemble classes (all variants)
├── single_trainers/
│   ├── classic_training.py      # RNNTrainer (single RNN/LSTM)
│   └── hcnn_training.py         # HCNNTrainer variants (Vanilla, PTF, LSpa, LForm)
├── ensemble_trainers/
│   ├── classic_training.py      # ClassicEnsembleTrainer
│   └── hcnn_training.py         # HCNNEnsembleTrainer variants
└── model_utils/
    └── custom_losses.py         # Custom loss functions (Log-cosh, MSE, etc.)

chaotic_data/                    # Chaotic system generators
├── systems.py                   # LorenzSolver, other dynamical systems
└── utils.py                     # Data utilities

data_utils/                       # Data preprocessing and utilities
├── preprocess.py                # Normalization, sliding windows, dataset classes
├── custom_fcts.py               # Custom functions
└── plottings.py                 # Plotting utilities

test/                             # Unit and integration tests
├── test_*.py                    # Model-specific tests (23 test files)
├── modules.py                   # Test support modules
└── *_comput_graph.py            # Computational graph visualizations

hcnns_chaos/                      # Alternative/legacy implementation structure
└── core/
    ├── base.py                  # Base classes and configuration
    ├── ensembles/               # Ensemble implementations
    └── cells.py                 # HCNN cell implementations
```

## Key Architecture Patterns

### 1. Model Hierarchy
All models (single) inherit from a common base and output named tuples:
- `VanillaHCNNForwardOutput` / `HCNNpTFForwardOutput` / etc.
- Fields: `expectations`, `states`, `delta_terms`, `forecasts`, `future_states`

### 2. Trainer Pattern
Both `single_trainers/` and `ensemble_trainers/` follow similar interface:
- `train_only()`: Train on training data, save best model by training loss
- `train_validate()`: Use calibration + forecast windows for validation
- Generate result directories with: trained models (`.pth`), loss logs (`epoch_losses.json`), forecast CSVs

### 3. Data Pipeline
- **Chaotic Systems**: `chaotic_data.systems.LorenzSolver()` generates trajectories
- **Normalization**: `data_utils.preprocess.Normalization_Strategy()` scales/centers data
- **Batching**: `SlidingWindowDataset`, `SlidingWindowDataLoader` for sequence windows
- **Testing**: `contextwindow_testdata_generator()` creates calibration + forecast splits

### 4. Ensemble Design
Ensembles wrap N individual models and support:
- Aggregation methods: `mean`, `median`, `weighted_mean`
- Uncertainty quantification via prediction variance
- Individual model access: `ensemble.get_model(i)`, `ensemble.get_all_models()`
- Extended forecasting with `forecast_horizon` parameter

## Common Development Tasks

### Running Tests
```bash
# Run all tests
pytest test/

# Run specific test
pytest test/test_VanillaHCNN.py -v

# Run test class or function
pytest test/test_VanillaHCNN.py::TestVanillaHCNNModel -v
pytest test/test_VanillaHCNN.py::test_forward_pass -v
```

### Training a Single Model
```bash
python run_data_run_vanilla_model.py
```
The script demonstrates:
- Loading Lorenz data via `chaotic_data.systems.LorenzSolver()`
- Normalization with `Normalization_Strategy`
- Creating sliding windows with `SlidingWindowDataset`
- Training a Vanilla_Model or other HCNN variant

### Key Imports Pattern
```python
from src.models.HCNN import Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model
from src.ensembles.hcnns import VanillaHCNNEnsemble, PTFHCNNEnsemble, LFormHCNNEnsemble, LSpaHCNNEnsemble
from src.single_trainers.hcnn_training import HCNNTrainer
from src.ensemble_trainers.hcnn_training import HCNNEnsembleTrainer

from chaotic_data.systems import LorenzSolver
from data_utils.preprocess import Normalization_Strategy, SlidingWindowDataset, SlidingWindowDataLoader
```

### Model Instantiation Pattern
All HCNN models share similar constructor signatures:
```python
model = Vanilla_Model(
    n_obs=3,                           # Number of observed variables
    n_hid_vars=10,                     # Number of hidden variables
    s0_nature='random_',               # Initial state: 'random_' or other
    train_s0=False,                    # Whether to train initial state
    batch_size=20,
    forecast_horizon=500               # For forecasting
)

# Forward pass
predictions, states, delta_terms, forecasts, future_states = model(data_window)
```

### Ensemble Instantiation Pattern
```python
ensemble = VanillaHCNNEnsemble(
    n_ensemble=5,
    n_obs_vars=1,
    n_hid_vars=10,
    s0_nature='random_',
    train_s0=True
)

# Predictions with aggregation
output = ensemble(data_window, aggregation_method="mean")

# Predictions with uncertainty
result = ensemble.predict_with_uncertainty(data_window, return_individual=True)
print(result['predictions_mean'], result['predictions_var'])
```

## HCNN Variants Overview

| Variant | Use Case | Key Feature |
|---------|----------|------------|
| **Vanilla** | Baseline | Simplest historical consistent formulation |
| **PTF** | Noisy data | Partial Teacher Forcing + dropout regularization |
| **LForm** | Long-term deps | LSTM-inspired gating for improved memory |
| **LSpa** | High-dim systems | Structured sparsity for scalability (100-200 vars) |

## Important Notes

### Device Handling
Code auto-detects and optimizes for:
- NVIDIA CUDA GPUs: `torch.cuda.is_available()`
- Apple Metal (MPS): `torch.backends.mps.is_available()`
- CPU fallback

Check device initialization in trainer and model code if performance issues arise.

### Data Shapes Convention
- Input: `(batch_size, sequence_length, n_obs)` - Observed variables only
- Hidden state: `(batch_size, n_hid_vars)` during forward pass
- Full state (internal): `(batch_size, n_obs + n_hid_vars)`

### Test File Organization
Test files in `test/` directory include:
- Core model tests: `test_VanillaHCNN.py`, `test_HCNNpTF.py`, `test_HCNNLForm.py`, `test_HCNNLSpa.py`
- Module tests: `test_vanilla_hcnncell.py`, `test_ptf_hcnncell.py`, `test_sparsity_modules.py`
- Integration tests: `test_use_diagonal_matrix.py`, `test_lstmformulation.py`
- Computational graph visualizations: `vanillaCell_comput_graph.py`, etc.

Tests use pytest framework with named tuples for assertion clarity.

### Result Artifacts
Training runs generate:
- `result_dir/best_model.pth` - Best checkpoint
- `result_dir/epoch_losses.json` - Per-epoch training/validation loss
- `result_dir/forecast_results_epoch_*.csv` - Predictions vs ground truth

### Legacy Code
`hcnns_chaos/` contains alternative implementations and may be phased out. Prefer `src/` for new development.

## Dependencies

Key packages (see `requirements.txt`):
- PyTorch 2.0.1+ (with torch, torchaudio, torchvision)
- NumPy, SciPy, Pandas
- Matplotlib, Plotly for visualization
- Jupyter for notebook workflows
- scikit-learn for utilities

## Jupyter Notebooks

Workflow notebooks in root directory:
- `workflow_hcnns_ensembles.ipynb` - HCNN ensemble experiments
- `workflow_rnn_lstm_fully_observables.ipynb` - RNN/LSTM baseline
- `workflow_rnns_lstms_ensembles.ipynb` - Classic ensemble training
- `workfow_hcnn_and_variants_fully_obsvervables.ipynb` - All HCNN variants
