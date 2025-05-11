import torch
import os
from typing import List, Optional, Literal, Tuple
from torch import nn

class RNNEnsemble:
    """
    Ensemble wrapper for managing multiple RNN_Model instances.

    Enables ensemble-based forecasting, full-sequence forward passes,
    and checkpoint management.

    Parameters
    ----------
    models : List[nn.Module]
        A list of RNN_Model instances.

    aggregation : Literal["mean", "median"]
        Strategy to aggregate outputs from all models.
    """

    def __init__(self, n_ensemble: int, input_size: int, hidden_size: int,
                 output_size: int, s0_nature: str, train_s0: bool,
                 init_range: Tuple[float, float], ext_vars: Optional[int] = None,
                 n_layers: int=1 , optimizer:Literal['adam', 'sgd'] = 'adam',\
                    learning_rate: float = 1e-4):
        
        """
        Initializes the ensemble with a list of models.
        """
        self.n_ensemble = n_ensemble
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.optimizer_type = optimizer
        if ext_vars is not None:
            self.ext_vars = ext_vars
            self.input_size = input_size + ext_vars
        else:
            self.ext_vars = None
            self.input_size = input_size

        self.models = []
        for member in range(self.n_ensemble):
            model = RNN_Model(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              output_size=self.output_size,
                              s0_nature=s0_nature,
                              train_s0=train_s0,
                              init_range=init_range,
                              num_layers=n_layers,
                              ext_vars=ext_vars)
            self.learning_rate = learning_rate
            if self.optimizer_type == 'adam':
                
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            elif optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

            
            self.models.append({f"member_{member+1}": model, 
                                f"optimizer_{member+1}": optimizer})
                
            





    def forward(self,
                input_sequence: torch.Tensor,
                initial_states: Optional[List[torch.Tensor]] = None,
                batch_of_externals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Runs full-sequence forward pass for all ensemble members and aggregates output.

        Parameters
        ----------
        input_sequence : torch.Tensor
            Shape: (batch, seq_len, input_size)

        initial_states : Optional[List[torch.Tensor]]
            List of initial hidden states per model.
            Each tensor has shape (num_layers, batch, hidden_size)

        batch_of_externals : Optional[torch.Tensor]
            External features, shape: (batch, seq_len, ext_vars)

        return_all : bool
            If True, returns individual outputs for all models.
            Else, returns the aggregated result.

        Returns
        -------
        torch.Tensor
            - return_all: List of RNNForwardOutput
        """
        all_outputs = []

        for i, model in enumerate(self.models):
            # model.eval()
            # with torch.no_grad():
            h0 = None if initial_states is None else initial_states[i]
            outputs_forward = model[f"model_{i+1}"].forward(
                input_sequence=input_sequence,
                initial_states=h0,
                batch_of_externals=batch_of_externals if batch_of_externals is not None else None
            )
            all_outputs.append(outputs_forward)

        return all_outputs


    def forecast(self,
                 calibration_window: torch.Tensor,
                 n_steps: int,
                 externals_for_calibration: Optional[torch.Tensor] = None,
                 externals_for_forecasts: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """
        Runs ensemble forecasting and aggregates predictions.

        Parameters
        ----------
        calibration_window : torch.Tensor
            Input sequence for calibration, shape: (seq_len, input_size)

        n_steps : int
            Number of forecast steps.

        externals_for_calibration : Optional[torch.Tensor]
            External vars during calibration, shape: (seq_len, ext_vars)

        externals_for_forecasts : Optional[torch.Tensor]
            External vars during forecasting, shape: (n_steps, ext_vars)

        return_all : bool
            Whether to return all individual model forecasts.

        Returns
        -------
        torch.Tensor
            Aggregated forecast (n_steps, output_size) or
            All forecasts (num_models, n_steps, output_size)
        """
        all_forecasts = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                _, forecast = model.forecast_seq_to_seq(
                    calibration_window.to(self.device),
                    n_steps,
                    externals_for_calibration if externals_for_calibration is not None else None,
                    externals_for_forecasts if externals_for_forecasts is not None else None,
                )
                all_forecasts.append(forecast.cpu())

        stacked_forecasts = torch.stack(all_forecasts)  # [num_models, n_steps, output_size]

        # if return_all:
        return stacked_forecasts


    def save_checkpoints(self, epoch: int, loss_list: List[float],
                         optimizers: List[torch.optim.Optimizer],
                         checkpoint_dir: str = "checkpoints", cleanup: bool = False, add_stuffs:Optional[str]="") -> None:
        """
        Save checkpoints for each model in the ensemble.
        """
        for i, model in enumerate(self.models):
            model.save_checkpoint(
                epoch=epoch,
                loss=loss_list[i],
                optimizer=optimizers[i],
                checkpoint_dir=checkpoint_dir,
                cleanup=cleanup,
                add_stuffs=add_stuffs
            )

    def load_checkpoints(self, checkpoint_paths: List[str],
                         optimizers: Optional[List[torch.optim.Optimizer]] = None) -> List[Tuple[int, float]]:
        """
        Load checkpoints into ensemble models.

        Returns
        -------
        List[Tuple[int, float]]
            List of (epoch, loss) for each restored model.
        """
        restore_info = []

        for i, path in enumerate(checkpoint_paths):
            optimizer = optimizers[i] if optimizers is not None else None
            epoch, loss = self.models[i].load_checkpoint(path, optimizer)
            restore_info.append((epoch, loss))

        return restore_info
    

    def load_ensemble_checkpoints(self, checkpoint_paths: List[str]) -> List[dict]:
        """
        Loads model and optimizer states from checkpoint files and returns the updated ensemble structure.

        Parameters
        ----------
        checkpoint_paths : List[str]
            List of file paths, one for each ensemble member.

        Returns
        -------
        List[dict]
            A list of dictionaries each containing:
            {
                "member_{i}": <RNN_Model>,
                "optimizer_{i}": <torch.optim.Optimizer>
            }
        """

        checkpoint_paths = [p for p in checkpoint_paths if not p.lower().endswith('.csv')]
        checkpoint_paths = [p for p in checkpoint_paths if not p.lower().endswith('.json')]
        if len(checkpoint_paths) != self.n_ensemble:
            raise ValueError(f"Expected {self.n_ensemble} checkpoint paths, got {len(checkpoint_paths)}")

        loaded_models = []

        for idx, checkpoint_path in enumerate(checkpoint_paths):
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"❌ Checkpoint not found at: {checkpoint_path}")

            # Reconstruct a fresh model instance
            model = RNN_Model(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            output_size=self.output_size,
                            s0_nature=self.s0_nature,
                            train_s0=self.train_s0,
                            init_range=self.init_range,
                            num_layers=self.n_layers,
                            ext_vars=self.ext_vars)

            # Create optimizer
            if self.optimizer_type == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=model.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print(f"✅ Loaded Model {idx+1} from {checkpoint_path} | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.6f}")

            # Append to list with proper structure
            loaded_models.append({
                f"member_{idx+1}": model,
                f"optimizer_{idx+1}": optimizer
            })

        return loaded_models

