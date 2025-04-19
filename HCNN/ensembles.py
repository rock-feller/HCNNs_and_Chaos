import pandas as pd
import torch , os, csv
import matplotlib.pyplot as plt
from .utils import  custom_fcts
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Optional, Literal
from glob import glob
from .models import Vanilla_Model , LSpa_Model , LForm_Model, PTF_Model
from datetime import datetime

class VanillaHCNNEnsembleTrainer:
    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        n_ext_vars: Optional[int] = None,
        s0_nature: str = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        optimizer: Literal ["sgd", "adam"] = "adam",
        lr: float = 1e-4,
        loss_fn: Literal["mse", "logcosh"]   =  "mse" ,
        save_dir: Optional[str] = "./checkpoints_vanilla_ensemble",
        best_on: Literal["median", "individual"] = "individual",
    ):
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.optimizer = optimizer
        self.n_ensemble = n_ensemble
        self.model_args = dict(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            n_ext_vars =  n_ext_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range
        )
        self.lr = lr
        self.loss_fn = loss_fn
        if self.loss_fn.lower() =="mse":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = custom_fcts.LogCoshLoss()
            
        # self.loss_fn = nn.MSELoss() if loss_fn == "mse" else nn.LogCoshLoss()
        self.save_dir = save_dir + f"_{best_on}_{timestamp}"
        self.models: List[Vanilla_Model] = []
        self.loss_histories: List[List[float]] = []
        self.best_losses: List[float] = [float('inf')] * n_ensemble
        self.best_on = best_on

        os.makedirs(self.save_dir, exist_ok=True)

        for i in range(n_ensemble):
            model = Vanilla_Model(**self.model_args)
            model.name = f"ensemble_member_{i+1}:{model._generate_model_name()}"  # for unique checkpoint names
            self.models.append(model)

    def train_only(
        self,
        data_loader,
        epochs: int = 10,
        verbose: bool = True, cleanup: bool = True
    ):
        if self.optimizer.lower() == "sgd":
            optimizers = [torch.optim.SGD(model.parameters(), lr=self.lr) for model in self.models]
        elif self.optimizer.lower() == "adam":
            optimizers = [torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.models]
        self.loss_histories = [[] for _ in range(self.n_ensemble)]
        best_median_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training Ensemble", position=0):
            epoch_losses = []

            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                epoch_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    results = model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(data_loader)
                self.loss_histories[i].append(avg_loss)
                epoch_losses.append(avg_loss)

                if self.best_on == "individual" and avg_loss < self.best_losses[i]:
                    model.save_checkpoint(epoch=epoch + 1, loss=avg_loss, optimizer=optimizer, checkpoint_dir=self.save_dir ,  cleanup=cleanup)
                    self.best_losses[i] = avg_loss


                if verbose:
                    print(f"[Ensemble Member {i + 1} out of {self.n_ensemble}] Epoch {epoch + 1} - Loss: {avg_loss:.6f}")

            if self.best_on == "median":
                median_loss = float(torch.tensor(epoch_losses).median())
                if median_loss < best_median_loss:
                    best_median_loss = median_loss
                    for i, model, optimizer, loss in zip(range(self.n_ensemble), self.models, optimizers, epoch_losses):
                        model.save_checkpoint(epoch=epoch + 1, loss=loss, optimizer=optimizer, checkpoint_dir=self.save_dir,  cleanup=True)
                    if verbose:
                        print(f"🟢 Epoch {epoch + 1}: New best median loss: {best_median_loss:.6f} — all models saved.")

    def forecast(self, context_data: torch.Tensor, forecast_window_length: int) -> List[torch.Tensor]:
        forecasts = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                result = model.forward(
                    data_window=context_data.unsqueeze(0),
                    forecast_horizon=forecast_window_length
                )
                forecasts.append(result.forecasts.squeeze(0))
        return forecasts

    def load_all_best(self, checkpoint_dir: Optional[str] = None):
        """
        Loads the latest checkpoint for each model in the ensemble based on naming convention.
        """
        checkpoint_dir = checkpoint_dir or self.save_dir

        for i, model in enumerate(self.models):
            pattern = os.path.join(checkpoint_dir, f"{model.name}_epoch*.pth")
            matching_files = glob(pattern)
            if not matching_files:
                print(f"No checkpoint found for {model.name}")
                continue
            latest = sorted(matching_files, key=os.path.getmtime)[-1]
            model.load_checkpoint(latest)

    def save_loss_history(self, path: str = "ensemble_loss_log.csv"):
        """
        Saves the loss history for each model as a CSV file.
        Each row: epoch, loss_model_0, loss_model_1, ...
        """
        max_epochs = max(len(hist) for hist in self.loss_histories)
        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            header = ["epoch"] + [f"loss_model_{i}" for i in range(self.n_ensemble)]
            writer.writerow(header)

            for epoch in range(max_epochs):
                row = [epoch + 1]
                for i in range(self.n_ensemble):
                    loss = self.loss_histories[i][epoch] if epoch < len(self.loss_histories[i]) else ''
                    row.append(loss)
                writer.writerow(row)
        print(f"Loss history saved to {path}")

    def train_and_validate(
        self,
        data_loader,
        calibration_data: torch.Tensor,
        validation_data: torch.Tensor,
        epochs: int = 10,
        verbose: bool = True,
        cleanup: bool = True
    ):
        forecast_horizon = validation_data.shape[0]

        if self.optimizer == "adam":
            optimizers = [torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models]
        else:
            optimizers = [torch.optim.SGD(m.parameters(), lr=self.lr) for m in self.models]

        self.loss_histories = [[] for _ in range(self.n_ensemble)]
        best_median_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training with Validation"):
            epoch_train_losses = []
            epoch_val_losses = []

            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                # === Training Phase ===
                model.train()
                train_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    results = model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(data_loader)
                self.loss_histories[i].append(avg_train_loss)
                epoch_train_losses.append(avg_train_loss)

                # === Validation Phase ===
                model.eval()
                with torch.no_grad():
                    results = model.forward(
                        data_window=calibration_data.unsqueeze(0),
                        forecast_horizon=forecast_horizon
                    )
                    forecast = results.forecasts.squeeze(0)
                    val_loss = self.loss_fn(forecast, validation_data).item()
                    epoch_val_losses.append(val_loss)

                if verbose:
                    print(f"[Member {i+1}] Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

                # Save per-member if best
                if self.best_on == "individual" and val_loss < self.best_losses[i]:
                    model.save_checkpoint(epoch=epoch + 1, loss=val_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=cleanup)
                    self.best_losses[i] = val_loss

            # Save ensemble if best median
            if self.best_on == "median":
                median_val_loss = float(torch.tensor(epoch_val_losses).median())
                if median_val_loss < best_median_loss:
                    best_median_loss = median_val_loss
                    for i, model, optimizer, val_loss in zip(range(self.n_ensemble), self.models, optimizers, epoch_val_losses):
                        model.save_checkpoint(epoch=epoch + 1, loss=val_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=True)
                    if verbose:
                        print(f"🟢 Epoch {epoch+1}: Best median validation loss improved to {median_val_loss:.6f}")





class PTFHCNNEnsembleTrainer:
    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        n_ext_vars: Optional[int] = None,
        s0_nature: str = 'random_',
        train_s0: bool = True,
        target_prob: float = 0.35,
        drop_output: bool = False,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        optimizer: Literal["adam", "sgd"] = "adam",
        lr: float = 1e-4,
        loss_fn: Literal["mse", "logcosh"] = "mse",
        save_dir: Optional[str] = "./checkpoints_ptf",
        best_on: Literal["median", "individual"] = "individual",
    ):
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.save_dir = f"{save_dir}_{best_on}_{timestamp}"
        self.optimizer = optimizer
        self.lr = lr
        self.best_on = best_on
        self.n_ensemble = n_ensemble

        self.loss_fn = nn.MSELoss() if loss_fn == "mse" else custom_fcts.LogCoshLoss()
        self.loss_histories = [[] for _ in range(n_ensemble)]
        self.best_losses = [float("inf")] * n_ensemble
        self.models: List[PTF_Model] = []

        os.makedirs(self.save_dir, exist_ok=True)

        self.model_args = dict(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            n_ext_vars=n_ext_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            target_prob=target_prob,
            drop_output=drop_output,
            init_range=init_range
        )

        for i in range(n_ensemble):
            model = PTF_Model(**self.model_args)
            model.name = f"ensemble_member_{i+1}:{model._generate_model_name()}"
            self.models.append(model)

    def train_only(self, data_loader, epochs=10, verbose=True, cleanup=True):
        optimizers = [
            torch.optim.Adam(m.parameters(), lr=self.lr) if self.optimizer == "adam"
            else torch.optim.SGD(m.parameters(), lr=self.lr)
            for m in self.models
        ]

        best_median_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training Ensemble"):
            epoch_losses = []

            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                epoch_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    results = model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(data_loader)
                self.loss_histories[i].append(avg_loss)
                epoch_losses.append(avg_loss)

                if self.best_on == "individual" and avg_loss < self.best_losses[i]:
                    model.save_checkpoint(epoch=epoch+1, loss=avg_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=cleanup)
                    self.best_losses[i] = avg_loss

                if verbose:
                    print(f"[PTF Member {i+1}] Epoch {epoch+1} - Loss: {avg_loss:.6f}")

            if self.best_on == "median":
                median_loss = float(torch.tensor(epoch_losses).median())
                if median_loss < best_median_loss:
                    best_median_loss = median_loss
                    for i, model, optimizer, loss in zip(range(self.n_ensemble), self.models, optimizers, epoch_losses):
                        model.save_checkpoint(epoch=epoch+1, loss=loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=True)

    def forecast(self, context_data: torch.Tensor, forecast_window_length: int):
        return [
            model.forward(data_window=context_data.unsqueeze(0), forecast_horizon=forecast_window_length).forecasts.squeeze(0)
            for model in self.models
        ]

    def load_all_best(self, checkpoint_dir: Optional[str] = None):
        checkpoint_dir = checkpoint_dir or self.save_dir
        for model in self.models:
            pattern = os.path.join(checkpoint_dir, f"{model.name}_epoch*.pth")
            files = sorted(glob(pattern), key=os.path.getmtime)
            if files:
                model.load_checkpoint(files[-1])

    def save_loss_history(self, path="ptf_loss_log.csv"):
        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + [f"loss_model_{i}" for i in range(self.n_ensemble)])
            max_epochs = max(len(h) for h in self.loss_histories)
            for epoch in range(max_epochs):
                row = [epoch + 1] + [self.loss_histories[i][epoch] if epoch < len(self.loss_histories[i]) else '' for i in range(self.n_ensemble)]
                writer.writerow(row)


    def train_and_validate(
        self,
        data_loader,
        calibration_data: torch.Tensor,
        validation_data: torch.Tensor,
        epochs: int = 10,
        verbose: bool = True,
        cleanup: bool = True
    ):
        forecast_horizon = validation_data.shape[0]

        if self.optimizer == "adam":
            optimizers = [torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models]
        else:
            optimizers = [torch.optim.SGD(m.parameters(), lr=self.lr) for m in self.models]

        self.loss_histories = [[] for _ in range(self.n_ensemble)]
        best_median_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training with Validation"):
            epoch_train_losses = []
            epoch_val_losses = []

            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                # === Training Phase ===
                model.train()
                train_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    results = model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(data_loader)
                self.loss_histories[i].append(avg_train_loss)
                epoch_train_losses.append(avg_train_loss)

                # === Validation Phase ===
                model.eval()
                with torch.no_grad():
                    results = model.forward(
                        data_window=calibration_data.unsqueeze(0),
                        forecast_horizon=forecast_horizon
                    )
                    forecast = results.forecasts.squeeze(0)
                    val_loss = self.loss_fn(forecast, validation_data).item()
                    epoch_val_losses.append(val_loss)

                if verbose:
                    print(f"[Member {i+1}] Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

                # Save per-member if best
                if self.best_on == "individual" and val_loss < self.best_losses[i]:
                    model.save_checkpoint(epoch=epoch + 1, loss=val_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=cleanup)
                    self.best_losses[i] = val_loss

            # Save ensemble if best median
            if self.best_on == "median":
                median_val_loss = float(torch.tensor(epoch_val_losses).median())
                if median_val_loss < best_median_loss:
                    best_median_loss = median_val_loss
                    for i, model, optimizer, val_loss in zip(range(self.n_ensemble), self.models, optimizers, epoch_val_losses):
                        model.save_checkpoint(epoch=epoch + 1, loss=val_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=True)
                    if verbose:
                        print(f"🟢 Epoch {epoch+1}: Best median validation loss improved to {median_val_loss:.6f}")



class HCNNLFormEnsembleTrainer:
    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        n_ext_vars: Optional[int] = None,
        s0_nature: str = 'random_',
        train_s0: bool = False,
        init_range: Tuple[float, float] = (-0.25, 0.25),
        init_diag: float = 1.0,
        optimizer: Literal["adam", "sgd"] = "adam",
        lr: float = 1e-4,
        loss_fn: Literal["mse", "logcosh"] = "mse",
        save_dir: Optional[str] = "./checkpoints_lform",
        best_on: Literal["median", "individual"] = "individual",
    ):
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.save_dir = f"{save_dir}_{best_on}_{timestamp}"
        self.optimizer = optimizer
        self.lr = lr
        self.best_on = best_on
        self.n_ensemble = n_ensemble

        self.loss_fn = nn.MSELoss() if loss_fn == "mse" else custom_fcts.LogCoshLoss()
        self.loss_histories = [[] for _ in range(n_ensemble)]
        self.best_losses = [float("inf")] * n_ensemble
        self.models: List[LForm_Model] = []

        os.makedirs(self.save_dir, exist_ok=True)

        self.model_args = dict(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            n_ext_vars=n_ext_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            init_diag=init_diag
        )

        for i in range(n_ensemble):
            model = LForm_Model(**self.model_args)
            model.name = f"ensemble_member_{i+1}:{model._generate_model_name()}"
            self.models.append(model)

    def train_only(self, data_loader, epochs: int = 10, verbose: bool = True, cleanup: bool = True):
        optimizers = [
            torch.optim.Adam(m.parameters(), lr=self.lr) if self.optimizer == "adam"
            else torch.optim.SGD(m.parameters(), lr=self.lr)
            for m in self.models
        ]

        best_median_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training Ensemble"):
            epoch_losses = []

            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                epoch_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    results = model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(data_loader)
                self.loss_histories[i].append(avg_loss)
                epoch_losses.append(avg_loss)

                if self.best_on == "individual" and avg_loss < self.best_losses[i]:
                    model.save_checkpoint(epoch=epoch+1, loss=avg_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=cleanup)
                    self.best_losses[i] = avg_loss

                if verbose:
                    print(f"[LForm Member {i+1}] Epoch {epoch+1} - Loss: {avg_loss:.6f}")

            if self.best_on == "median":
                median_loss = float(torch.tensor(epoch_losses).median())
                if median_loss < best_median_loss:
                    best_median_loss = median_loss
                    for i, model, optimizer, loss in zip(range(self.n_ensemble), self.models, optimizers, epoch_losses):
                        model.save_checkpoint(epoch=epoch+1, loss=loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=True)

    def forecast(self, context_data: torch.Tensor, forecast_window_length: int):
        return [
            model.forward(data_window=context_data.unsqueeze(0), forecast_horizon=forecast_window_length).forecasts.squeeze(0)
            for model in self.models
        ]

    def load_all_best(self, checkpoint_dir: Optional[str] = None):
        checkpoint_dir = checkpoint_dir or self.save_dir
        for model in self.models:
            pattern = os.path.join(checkpoint_dir, f"{model.name}_epoch*.pth")
            files = sorted(glob(pattern), key=os.path.getmtime)
            if files:
                model.load_checkpoint(files[-1])

    def save_loss_history(self, path="lform_loss_log.csv"):
        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + [f"loss_model_{i}" for i in range(self.n_ensemble)])
            max_epochs = max(len(h) for h in self.loss_histories)
            for epoch in range(max_epochs):
                row = [epoch + 1] + [self.loss_histories[i][epoch] if epoch < len(self.loss_histories[i]) else '' for i in range(self.n_ensemble)]
                writer.writerow(row)


    def train_and_validate(
        self,
        data_loader,
        calibration_data: torch.Tensor,
        validation_data: torch.Tensor,
        epochs: int = 10,
        verbose: bool = True,
        cleanup: bool = True
    ):
        forecast_horizon = validation_data.shape[0]

        if self.optimizer == "adam":
            optimizers = [torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models]
        else:
            optimizers = [torch.optim.SGD(m.parameters(), lr=self.lr) for m in self.models]

        self.loss_histories = [[] for _ in range(self.n_ensemble)]
        best_median_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training with Validation"):
            epoch_train_losses = []
            epoch_val_losses = []

            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                # === Training Phase ===
                model.train()
                train_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    results = model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(data_loader)
                self.loss_histories[i].append(avg_train_loss)
                epoch_train_losses.append(avg_train_loss)

                # === Validation Phase ===
                model.eval()
                with torch.no_grad():
                    results = model.forward(
                        data_window=calibration_data.unsqueeze(0),
                        forecast_horizon=forecast_horizon
                    )
                    forecast = results.forecasts.squeeze(0)
                    val_loss = self.loss_fn(forecast, validation_data).item()
                    epoch_val_losses.append(val_loss)

                if verbose:
                    print(f"[Member {i+1}] Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

                # Save per-member if best
                if self.best_on == "individual" and val_loss < self.best_losses[i]:
                    model.save_checkpoint(epoch=epoch + 1, loss=val_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=cleanup)
                    self.best_losses[i] = val_loss

            # Save ensemble if best median
            if self.best_on == "median":
                median_val_loss = float(torch.tensor(epoch_val_losses).median())
                if median_val_loss < best_median_loss:
                    best_median_loss = median_val_loss
                    for i, model, optimizer, val_loss in zip(range(self.n_ensemble), self.models, optimizers, epoch_val_losses):
                        model.save_checkpoint(epoch=epoch + 1, loss=val_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=True)
                    if verbose:
                        print(f"🟢 Epoch {epoch+1}: Best median validation loss improved to {median_val_loss:.6f}")




class LSPaEnsembleTrainer:
    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        s0_nature: str = 'random_',
        train_s0: bool = True,
        n_ext_vars: Optional[int] = None,
        sparsity_ratio: float = 0.25,
        mask_type:  Literal["random_block", "non_obs_block"] = "non_obs_block",
        init_range: Tuple[float, float] = (-0.25, 0.25),
        optimizer: Literal["adam", "sgd"] = "adam",
        lr: float = 1e-4,
        loss_fn: Literal["mse", "logcosh"] = "mse",
        save_dir: Optional[str] = "./checkpoints_ensemble_lspa",
        best_on: Literal["median", "individual"] = "individual"):
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.save_dir = f"{save_dir}_{best_on}_{timestamp}"
        self.optimizer = optimizer
        self.lr = lr
        self.best_on = best_on
        self.n_ensemble = n_ensemble

        self.loss_fn = nn.MSELoss() if loss_fn == "mse" else custom_fcts.LogCoshLoss()
        self.loss_histories = [[] for _ in range(n_ensemble)]
        self.best_losses = [float("inf")] * n_ensemble
        self.models: List[LSpa_Model] = []

        os.makedirs(self.save_dir, exist_ok=True)

        self.model_args = dict(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            n_ext_vars=n_ext_vars,
            sparsity_ratio=sparsity_ratio,
            mask_type=mask_type,
            init_range=init_range
        )

        for i in range(n_ensemble):
            model = LSpa_Model(**self.model_args)
            model.name = f"ensemble_member_{i+1}:{model._generate_model_name()}"
            self.models.append(model)

    def train_only(self, data_loader, epochs: int = 10, verbose: bool = True, cleanup: bool = True):
        if self.optimizer == "adam":
            optimizers = [torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models]
        else:
            optimizers = [torch.optim.SGD(m.parameters(), lr=self.lr) for m in self.models]

        best_median_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training Ensemble"):
            epoch_losses = []

            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                epoch_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    results = model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(data_loader)
                self.loss_histories[i].append(avg_loss)
                epoch_losses.append(avg_loss)

                if self.best_on == "individual" and avg_loss < self.best_losses[i]:
                    model.save_checkpoint(epoch=epoch+1, loss=avg_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=cleanup)
                    self.best_losses[i] = avg_loss

                if verbose:
                    print(f"[LSpa Member {i+1}] Epoch {epoch+1} - Loss: {avg_loss:.6f}")

            if self.best_on == "median":
                median_loss = float(torch.tensor(epoch_losses).median())
                if median_loss < best_median_loss:
                    best_median_loss = median_loss
                    for i, model, optimizer, loss in zip(range(self.n_ensemble), self.models, optimizers, epoch_losses):
                        model.save_checkpoint(epoch=epoch+1, loss=loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=True)
                    if verbose:
                        print(f"🟢 Epoch {epoch+1}: Best median improved to {best_median_loss:.6f}")

    def forecast(self, context_data: torch.Tensor, forecast_window_length: int) -> List[torch.Tensor]:
        return [
            model.forward(data_window=context_data.unsqueeze(0), forecast_horizon=forecast_window_length).forecasts.squeeze(0)
            for model in self.models
        ]

    def load_all_best(self, checkpoint_dir: Optional[str] = None):
        checkpoint_dir = checkpoint_dir or self.save_dir
        for model in self.models:
            pattern = os.path.join(checkpoint_dir, f"{model.name}_epoch*.pth")
            files = sorted(glob(pattern), key=os.path.getmtime)
            if files:
                model.load_checkpoint(files[-1])

    def save_loss_history(self, path="lspa_loss_log.csv"):
        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + [f"loss_model_{i}" for i in range(self.n_ensemble)])
            max_epochs = max(len(h) for h in self.loss_histories)
            for epoch in range(max_epochs):
                row = [epoch + 1] + [self.loss_histories[i][epoch] if epoch < len(self.loss_histories[i]) else '' for i in range(self.n_ensemble)]
                writer.writerow(row)

    def train_and_validate(
        self,
        data_loader,
        calibration_data: torch.Tensor,
        validation_data: torch.Tensor,
        epochs: int = 10,
        verbose: bool = True,
        cleanup: bool = True
    ):
        forecast_horizon = validation_data.shape[0]

        if self.optimizer == "adam":
            optimizers = [torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models]
        else:
            optimizers = [torch.optim.SGD(m.parameters(), lr=self.lr) for m in self.models]

        self.loss_histories = [[] for _ in range(self.n_ensemble)]
        best_median_loss = float("inf")

        for epoch in tqdm(range(epochs), desc="Training with Validation"):
            epoch_train_losses = []
            epoch_val_losses = []

            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                # === Training Phase ===
                model.train()
                train_loss = 0.0
                for batch in data_loader:
                    optimizer.zero_grad()
                    results = model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(data_loader)
                self.loss_histories[i].append(avg_train_loss)
                epoch_train_losses.append(avg_train_loss)

                # === Validation Phase ===
                model.eval()
                with torch.no_grad():
                    results = model.forward(
                        data_window=calibration_data.unsqueeze(0),
                        forecast_horizon=forecast_horizon
                    )
                    forecast = results.forecasts.squeeze(0)
                    val_loss = self.loss_fn(forecast, validation_data).item()
                    epoch_val_losses.append(val_loss)

                if verbose:
                    print(f"[Member {i+1}] Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

                # Save per-member if best
                if self.best_on == "individual" and val_loss < self.best_losses[i]:
                    model.save_checkpoint(epoch=epoch + 1, loss=val_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=cleanup)
                    self.best_losses[i] = val_loss

            # Save ensemble if best median
            if self.best_on == "median":
                median_val_loss = float(torch.tensor(epoch_val_losses).median())
                if median_val_loss < best_median_loss:
                    best_median_loss = median_val_loss
                    for i, model, optimizer, val_loss in zip(range(self.n_ensemble), self.models, optimizers, epoch_val_losses):
                        model.save_checkpoint(epoch=epoch + 1, loss=val_loss, optimizer=optimizer, checkpoint_dir=self.save_dir, cleanup=True)
                    if verbose:
                        print(f"🟢 Epoch {epoch+1}: Best median validation loss improved to {median_val_loss:.6f}")

