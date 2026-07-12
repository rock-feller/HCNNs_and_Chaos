# HCNN Ensemble Implementations

This module provides comprehensive ensemble implementations for all HCNN variants, enabling improved prediction accuracy and uncertainty quantification through model averaging and diversity.

## Available Ensemble Classes

### 1. VanillaHCNNEnsemble
Ensemble of Vanilla HCNN models providing baseline ensemble performance.

**Key Features:**
- Simple, robust ensemble implementation
- Multiple aggregation methods (mean, median, weighted)
- Uncertainty quantification through prediction variance
- Efficient parallel training support

**Usage:**
```python
from hcnn.core.ensembles import VanillaHCNNEnsemble

ensemble = VanillaHCNNEnsemble(
    n_ensemble=5,
    n_obs_vars=1,
    n_hid_vars=10,
    s0_nature='random_',
    train_s0=True,
    init_range=(-0.5, 0.5)
)
```

### 2. PTFHCNNEnsemble
Ensemble of Partial Teacher Forcing HCNN models with advanced regularization.

**Key Features:**
- Dropout-based regularization across ensemble members
- Multiple dropout strategies (adaptive, linear, exponential, step, cosine)
- Enhanced convergence speed through teacher forcing
- Robust performance on noisy data

**Usage:**
```python
from hcnn.core.ensembles import PTFHCNNEnsemble

ensemble = PTFHCNNEnsemble(
    n_ensemble=5,
    n_obs_vars=1,
    n_hid_vars=10,
    s0_nature='random_',
    train_s0=True,
    target_prob=0.2,
    dropout_strategy='adaptive'
)
```

### 3. LFormHCNNEnsemble
Ensemble of LSTM Formulation HCNN models optimized for sequential processing.

**Key Features:**
- LSTM-inspired gating mechanisms
- Superior long-term dependency modeling
- Enhanced memory retention capabilities
- Optimized for time series with complex temporal patterns

**Usage:**
```python
from hcnn.core.ensembles import LFormHCNNEnsemble

ensemble = LFormHCNNEnsemble(
    n_ensemble=5,
    n_obs_vars=1,
    n_hid_vars=10,
    s0_nature='random_',
    train_s0=True
)
```

### 4. LSpaHCNNEnsemble
Ensemble of Large Sparse HCNN models designed for scalability.

**Key Features:**
- Structured sparsity for computational efficiency
- Scalable to high-dimensional systems (100-200 variables)
- Multiple sparsity patterns (random_block, structured)
- Memory-efficient implementation

**Usage:**
```python
from hcnn.core.ensembles import LSpaHCNNEnsemble

ensemble = LSpaHCNNEnsemble(
    n_ensemble=5,
    n_obs_vars=1,
    n_hid_vars=10,
    s0_nature='random_',
    train_s0=True,
    sparsity_ratio=0.5,
    mask_type='random_block'
)
```

## Aggregation Methods

All ensemble classes support multiple aggregation strategies:

### Mean Aggregation (Default)
```python
output = ensemble(data_window, aggregation_method="mean")
```
- Simple arithmetic mean of all model predictions
- Provides smooth, stable predictions
- Reduces prediction variance

### Median Aggregation
```python
output = ensemble(data_window, aggregation_method="median")
```
- Robust to outlier predictions
- Better for handling model failures
- More conservative predictions

### Weighted Mean Aggregation
```python
output = ensemble(data_window, aggregation_method="weighted_mean")
```
- Weights based on individual model performance
- Automatically adapts to model quality
- Requires training history for weight calculation

## Advanced Features

### Uncertainty Quantification
All ensembles provide prediction uncertainty through variance calculation:

```python
# Get detailed predictions with uncertainty
result = ensemble.predict_with_uncertainty(
    data_window=test_data,
    return_individual=True
)

# Access uncertainty metrics
prediction_mean = result["predictions_mean"]
prediction_var = result["predictions_var"]
individual_predictions = result["individual_predictions"]
```

### Individual Model Access
Access specific models within the ensemble:

```python
# Get specific model
model_0 = ensemble.get_model(0)

# Train individual model
individual_output = model_0(data_window)

# Get all models
all_models = ensemble.get_all_models()
```

### Forecasting with Ensembles
Extended forecasting capabilities:

```python
# Multi-step forecasting
forecast_output = ensemble(
    data_window=input_data,
    forecast_horizon=50,
    aggregation_method="mean"
)

# Access forecasts
future_predictions = forecast_output.forecasts
future_states = forecast_output.future_states
```

## Training Ensembles

### Basic Training
```python
from hcnn.utils.fully_unfolded_mode import FullyUnfoldedTrainer

trainer = FullyUnfoldedTrainer(
    model=ensemble,
    loss_fn="mse",
    optimizer_type="adam",
    learning_rate=1e-3,
    device=device
)

trainer.train_and_validate(
    training_data=train_data,
    num_epochs=100,
    calibration_window=calibration_data,
    val_data=validation_data
)
```

### Parallel Training
For faster training, individual models can be trained in parallel:

```python
# Train individual models separately
for i in range(ensemble.n_ensemble):
    model = ensemble.get_model(i)
    individual_trainer = FullyUnfoldedTrainer(model=model, ...)
    individual_trainer.train_only(training_data, num_epochs=100)
```

## Performance Guidelines

### Ensemble Size Recommendations
- **Small systems (1-5 variables)**: 3-5 models
- **Medium systems (5-20 variables)**: 5-10 models  
- **Large systems (20+ variables)**: 10-20 models

### Memory Considerations
- Memory usage scales linearly with ensemble size
- LSpa ensembles use ~50% less memory due to sparsity
- Consider gradient checkpointing for very large ensembles

### Computational Efficiency
- Training time scales linearly with ensemble size
- Inference time: ~N times single model (where N = ensemble size)
- PTF ensembles have ~10-15% overhead due to dropout

## Best Practices

### 1. Diversity Promotion
```python
# Use different initialization ranges for diversity
ensemble_configs = []
for i in range(n_ensemble):
    config = {
        'init_range': (-0.5 + i*0.1, 0.5 + i*0.1),
        'learning_rate': 1e-3 * (0.8 + i*0.1)
    }
    ensemble_configs.append(config)
```

### 2. Validation Strategy
```python
# Use different validation splits for each model
for i, model in enumerate(ensemble.get_all_models()):
    val_start = i * (len(val_data) // n_ensemble)
    val_end = (i + 1) * (len(val_data) // n_ensemble)
    model_val_data = val_data[val_start:val_end]
    # Train with model-specific validation set
```

### 3. Early Stopping
```python
# Implement ensemble-aware early stopping
class EnsembleEarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_losses = [float('inf')] * n_ensemble
        self.wait_counts = [0] * n_ensemble
    
    def should_stop(self, ensemble_losses):
        # Check if majority of models should stop
        stop_count = 0
        for i, loss in enumerate(ensemble_losses):
            if loss < self.best_losses[i] - self.min_delta:
                self.best_losses[i] = loss
                self.wait_counts[i] = 0
            else:
                self.wait_counts[i] += 1
            
            if self.wait_counts[i] >= self.patience:
                stop_count += 1
        
        return stop_count > len(ensemble_losses) // 2
```

## Example: Complete Workflow

```python
import torch
from hcnn.core.ensembles import VanillaHCNNEnsemble
from hcnn.utils.fully_unfolded_mode import FullyUnfoldedTrainer

# 1. Create ensemble
ensemble = VanillaHCNNEnsemble(
    n_ensemble=5,
    n_obs_vars=1,
    n_hid_vars=10,
    s0_nature='random_',
    train_s0=True
)

# 2. Train ensemble
trainer = FullyUnfoldedTrainer(
    model=ensemble,
    loss_fn="mse",
    optimizer_type="adam",
    learning_rate=1e-3
)

trainer.train_and_validate(
    training_data=train_data,
    num_epochs=100,
    calibration_window=calibration_data,
    val_data=validation_data
)

# 3. Evaluate with uncertainty
result = ensemble.predict_with_uncertainty(
    data_window=test_data,
    return_individual=True
)

# 4. Analyze results
print(f"Mean prediction: {result['predictions_mean'].mean():.6f}")
print(f"Prediction uncertainty: {result['predictions_var'].mean():.6f}")
print(f"Individual model agreement: {result['model_agreement']:.3f}")
```

This comprehensive ensemble framework provides robust, scalable solutions for chaotic system modeling with quantified uncertainty and improved prediction accuracy.
