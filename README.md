# HCNNs\_and\_Chaos

In this repo, you will find all the codes used in experiments from my PhD, focusing on the modeling of chaotic systems using Historical Consistent Neural Networks (HCNNs) and Recurrent Neural Networks (RNNs). The project includes support for both single-model and ensemble-based training pipelines implemented in PyTorch.

## Code Implementation Overview

The entire modeling stack has been restructured under a unified `src/` directory that houses all components: models, training logic, utilities, and ensemble handling. This design provides modularity, reusability, and clean separation between HCNN and RNN workflows, whether run as single models or ensembles.

## 📁 Project Structure

```
src
├── data                         # Custom data generation and loading utilities
├── ensembles                   # All ensemble model definitions
│   ├── classics.py             # LSTM and RNN ensemble classes
│   ├── hcnns.py                # Vanilla, PTF, LForm, LSpa HCNN ensemble classes
├── ensemble_trainers           # Ensemble trainer implementations
│   ├── classic_training.py    # RNN and LSTM ensemble training
│   ├── hcnn_training.py       # HCNN ensemble training modules
├── models                      # Single model definitions
│   ├── classic.py             # RNN_Model and LSTM_Model implementations
│   ├── HCNN
│   │   ├── hcnn_models.py     # Vanilla, PTF, LSpa, and LForm HCNN variants
│   │   ├── modules.py         # Core HCNN module definitions
│   │   ├── utils/             # Custom layers, functions, and math utilities
│   │   ├── visualization.py   # Visualization support for HCNN behavior
├── model_utils                 # Shared utilities
│   ├── custom_losses.py       # Log-cosh, MSE and other loss functions
├── single_trainers             # Single-model trainer modules
│   ├── classic_training.py    # RNNTrainer for single RNN/LSTM
│   ├── hcnn_training.py       # Trainers for Vanilla, PTF, LSpa, LForm
```

## 🧠 Modeling Support

The framework supports both single-model and ensemble training for:

* **RNNs / LSTMs** via `models/classic.py` and `ensembles/classics.py`
* **HCNN Variants**:

  * `Vanilla_Model`
  * `PTF_Model` (Partial Teacher Forcing)
  * `LForm_Model` (LSTM Formulation of HCNN)
  * `LSpa_Model` (Large Sparse HCNN)

These are implemented under `models/HCNN/` and `ensembles/hcnns.py`, with modular training logic handled separately for single and ensemble variants.

## 🎯 Training Modes

Each model (RNN, LSTM, or HCNN) can be trained via its respective trainer:

* **Single-model training** (under `single_trainers/`):

  * `RNNTrainer` for `RNN_Model` or `LSTM_Model`
  * `HCNNTrainer` for Vanilla HCNN  and related variants

* **Ensemble training** (under `ensemble_trainers/`):

  * `ClassicEnsembleTrainer` for RNNs and LSTMs
  * `HCNNEnsembleTrainer` variants for HCNNs (Vanilla, PTF, LForm, LSpa)

Each trainer supports:

* `train_only`: Trains on training data and tracks best model based on training loss.
* `train_validate`: Uses calibration and forecast windows to validate predictions and save models based on best validation performance.

## 📊 Result Tracking

Every training run automatically generates a result directory containing:

* **Trained Models** (`.pth` files): Best performing checkpoint
* **Epoch Loss Logs** (`epoch_losses.json`): Per-epoch training/validation loss
* **Forecast Outputs** (`forecast_results_epoch_*.csv`): Predicted vs true values for each epoch

## ⚙️ Preprocessing & Utilities

The `data` and `data_prep` modules (now folded into `src/`) include:

* `SlidingWindowDataset`, `InpTarg_TSDataset` for batching
* `Normalization_Strategy` for scaling and centering
* `contextwindow_testdata_generator` for creating calibration + forecast sequences

## 🚀 GPU/MPS Optimization

The code is compatible with:

* **NVIDIA CUDA GPUs**
* **Apple M Series with Metal (MPS)**

Automatic device detection ensures optimal hardware acceleration for training and inference workflows.
