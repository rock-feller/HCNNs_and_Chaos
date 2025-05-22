import pandas as pd
import json
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import warnings
from ..model_utils.custom_losses import LogCoshLoss
from typing import List, Tuple, Optional,Dict, Literal
from datetime import datetime



class RNNTrainer:
    def __init__(self, model: nn.Module , 
                 loss_fn: Literal["mse", "logcosh"]   =  "mse" ,
                 backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch',
                 learning_rate: float = 1e-4,
                 variable_names: Optional[List[str]] = None,optimizer_type: Literal['adam', 'sgd'] = 'adam',):

        """
        Trainer for RNN_Model with support for forecasting and flexible gradient scheduling.
        """
        self.model = model
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) if self.optimizer_type == "adam" \
                          else torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.backprop_mode = backprop_mode
        # self.loss_fn = loss_fn
        self.loss_fn = nn.MSELoss() if loss_fn == "mse" else LogCoshLoss()
        self.device = model._get_default_device()
        self.variable_names = [f"var_{i+1}" for i in range(self.model.output_size)] if variable_names is None else variable_names


        
        save_dir = f"./checkpoints_{self.model._generate_model_name().split('_')[0]}"
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        self.save_dir = save_dir + f"_{timestamp}"

        os.makedirs(self.save_dir, exist_ok=True)

    def train_only(self, data_loader: DataLoader , num_epochs: int)-> Tuple[list, list]:
        """
        Trains the model and returns logs of average losses and forecasts.

        Returns
        -------
        Tuple[List[float], List[Tuple[torch.Tensor, torch.Tensor]]]
            - epoch_losses: List of average loss per epoch
            - forecast_results: List of (out_seq, forecast) per epoch


        Parameters
        ----------
        data_loader : DataLoader
            Yields batches of ([inputs+externals], [targets+externals]) if ext_vars is not None
            Yields batches of ([inputs], [targets]) if ext_vars is None


        backprop_mode : str
            'per_batch' - Backprop and step after each batch (default).
            'per_epoch' - Accumulate loss over all batches before backward.

        training_mode : str
            'train_only' : to save best model based on avg train loss per epoch
            'train_forecast' - to save best model based on validation loss per epoch.

        calibration_window : Optional[torch.Tensor]
            Calibration window for forecasting. Shape: (seq_len, input_size)
        
        val_data: Optional[torch.Tensor]
            Ground truth which is your validation sequence. Shape: (n_steps, output_size)
        """

        epoch_losses = []  



        if self.backprop_mode == 'per_batch':

            for epoch in tqdm(range(num_epochs), desc="Training Only"):  
                epoch_loss = 0.0
                accumulated_loss = 0.0
                batch_count = 0
                
                batch_losses = []
                self.model.train()

                for batch_inps, batch_outs in data_loader:

                    batch_size = batch_inps.size(0)
                    if self.model.name.split('_')[0] == 'LSTMModel':
                        # Expand initial hidden state
                        initial_h = self.model.initial_h.expand(-1, batch_size, -1).contiguous()
                        initial_c = self.model.initial_c.expand(-1, batch_size, -1).contiguous()
                        initial_state = (initial_h, initial_c)
        
                    elif self.model.name.split('_')[0] == 'RNNModel':


                        initial_state = self.model.initial_hidden_state.expand(-1, batch_size, -1).contiguous()


                    if self.model.ext_vars is not None:


                        outputs_forward = self.model.forward(input_sequence=batch_inps[: ,:,:- self.model.ext_vars], 
                                                    initial_states=initial_state, 
                                                    batch_of_externals = batch_inps[:,:, - self.model.ext_vars:])#, externals=externals)

                        batch_outs = batch_outs[:,:,: - self.model.ext_vars]

                        loss = self.loss_fn(outputs_forward.output_sequence, batch_outs)
                    
                    else:

                        outputs_forward = self.model.forward(input_sequence=batch_inps, initial_states=initial_state)#, externals=externals)



                    batch_loss = self.loss_fn(outputs_forward.output_sequence, batch_outs)
                
                    batch_count += 1

                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                    batch_losses.append(batch_loss.item())

                    current_best_batch_loss= self.save_best_model_based_on_train_loss(epoch=epoch, 
                                                                                train_loss=batch_loss, 
                                                                                save_path="best_train_avg_loss.pth")

                epoch_losses.append({'epoch': epoch+1,'avg_train_loss': sum(batch_losses)/batch_count})

                

                print(f"[End of Epoch {epoch+1}] Best train Loss: {current_best_batch_loss:.6f}")

            self.save_epochs_losses_to_json(epoch_losses=epoch_losses, filename="epoch_losses.json")

        if self.backprop_mode == 'per_epoch':

            for epoch in tqdm(range(num_epochs), desc="Training Only"):
                accumulated_loss = 0.0
                batch_count = 0
                
                batch_losses = []
                self.model.train()

                for batch_inps, batch_outs in data_loader:

                    batch_size = batch_inps.size(0)
                    if self.model.name.split('_')[0] == 'LSTMModel':
                        initial_h = self.model.initial_h.expand(-1, batch_size, -1).contiguous()
                        initial_c = self.model.initial_c.expand(-1, batch_size, -1).contiguous()
                        initial_state = (initial_h, initial_c)
        
                    elif self.model.name.split('_')[0] == 'RNNModel':


                        initial_state = self.model.initial_hidden_state.expand(-1, batch_size, -1).contiguous()


                    if self.model.ext_vars is not None:


                        outputs_forward = self.model.forward(input_sequence=batch_inps[: ,:,:- self.model.ext_vars], 
                                                    initial_states=initial_state, 
                                                    batch_of_externals = batch_inps[:,:, - self.model.ext_vars:])#, externals=externals)

                        batch_outs = batch_outs[:,:,: - self.model.ext_vars]

                        loss = self.loss_fn(outputs_forward.output_sequence, batch_outs)
                    
                    else:

                        outputs_forward = self.model.forward(input_sequence=batch_inps, initial_states=initial_state)#, externals=externals)



                    loss = self.loss_fn(outputs_forward.output_sequence, batch_outs)
                
                    batch_count += 1

                    accumulated_loss += loss


                self.optimizer.zero_grad()
                accumulated_loss.backward()
                self.optimizer.step()

                avg_train_loss = accumulated_loss.item()/batch_count

                forecasts =  None
                current_best_train_loss= self.save_best_model_based_on_train_loss(epoch=epoch, 
                                                                        train_loss=avg_train_loss ,
                                                                        save_path="best_train_avg_loss.pth")

                epoch_losses.append({'epoch': epoch+1,'avg_train_loss': avg_train_loss})



                print(f"[End of Epoch {epoch+1}] Best train Loss: {current_best_train_loss:.6f}")

            self.save_epochs_losses_to_json(epoch_losses=epoch_losses, filename="epoch_losses.json")

                    
    def train_validate(self, data_loader, calibration_window:torch.Tensor, 
                       val_data: torch.Tensor , num_epochs:int)-> Tuple[list, list]:
        """
        Trains the model and returns logs of average losses and forecasts.

        Returns
        -------
        Tuple[List[float], List[Tuple[torch.Tensor, torch.Tensor]]]
            - epoch_losses: List of average loss per epoch
            - forecast_results: List of (out_seq, forecast) per epoch


        Parameters
        ----------
        data_loader : DataLoader
            Yields batches of ([inputs+externals], [targets+externals]) if ext_vars is not None
            Yields batches of ([inputs], [targets]) if ext_vars is None


        calibration_window : Optional[torch.Tensor]
            Calibration window for forecasting. Shape: (seq_len, input_size)
        
        val_data: Optional[torch.Tensor]
            Ground truth which is your validation sequence. Shape: (n_steps, output_size)
        """

        epoch_losses = []  



        if self.backprop_mode == 'per_batch':

            for epoch in tqdm(range(num_epochs), desc="Training and validating"):  
                epoch_loss = 0.0
                accumulated_loss = 0.0
                batch_count = 0
                
                batch_losses = []
                self.model.train()

                for batch_inps, batch_outs in data_loader:

                    batch_size = batch_inps.size(0)
                    if self.model.name.split('_')[0] == 'LSTMModel':
                        initial_h = self.model.initial_h.expand(-1, batch_size, -1).contiguous()
                        initial_c = self.model.initial_c.expand(-1, batch_size, -1).contiguous()
                        initial_state = (initial_h, initial_c)
        
                    elif self.model.name.split('_')[0] == 'RNNModel':


                        initial_state = self.model.initial_hidden_state.expand(-1, batch_size, -1).contiguous()


                    if self.model.ext_vars is not None:


                        outputs_forward = self.model.forward(input_sequence=batch_inps[: ,:,:- self.model.ext_vars], 
                                                    initial_states=initial_state, 
                                                    batch_of_externals = batch_inps[:,:, - self.model.ext_vars:])#, externals=externals)

                        batch_outs = batch_outs[:,:,: - self.model.ext_vars]

                        loss = self.loss_fn(outputs_forward.output_sequence, batch_outs)
                    
                    else:

                        outputs_forward = self.model.forward(input_sequence=batch_inps, initial_states=initial_state)#, externals=externals)



                    batch_loss = self.loss_fn(outputs_forward.output_sequence, batch_outs)
                
                    batch_count += 1

                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                    batch_losses.append(batch_loss.item())

                    
                    self.model.eval() 

                    if self.model.ext_vars is not None:


                        
                        with torch.no_grad():

                            out_seq, forecasts = self.model.forecast_seq_to_seq(
                                calibration_window=calibration_window[:,:-self.model.ext_vars] , 
                                n_steps=val_data.shape[0],
                                externals_for_calibration=calibration_window[:,-self.model.ext_vars:] ,
                                externals_for_forecasts=val_data[:,-self.model.ext_vars:])
                            
                    else:
                        
                        with torch.no_grad():
                            out_seq, forecasts = self.model.forecast_seq_to_seq(
                                calibration_window=calibration_window, n_steps=val_data.shape[0])
                            



                    current_val_loss , best_epoch_val_loss = self.save_best_model_based_on_val_loss(epoch=epoch, 
                                                                                    out_seq=out_seq,
                                                                                    forecasts=forecasts,
                                                                                    val_data=val_data,
                                                                                save_path="best_val_avg_loss.pth")
                            
                    
                epoch_losses.append({'epoch': epoch+1,'train_loss': sum(batch_losses)/batch_count,
                                          'val_loss':current_val_loss})



                print(f"[End of Epoch {epoch+1}] Current val loss: {current_val_loss:.6f} || Best val Loss: {best_epoch_val_loss:.6f}")

            self.save_epochs_losses_to_json(epoch_losses=epoch_losses, filename="epoch_losses.json")



        if self.backprop_mode == 'per_epoch':

            for epoch in tqdm(range(num_epochs), desc="Training and validating"):  
                accumulated_loss = 0.0
                batch_count = 0
                
                self.model.train()

                for batch_inps, batch_outs in data_loader:

                    batch_size = batch_inps.size(0)
                    if self.model.name.split('_')[0] == 'LSTMModel':
                        initial_h = self.model.initial_h.expand(-1, batch_size, -1).contiguous()
                        initial_c = self.model.initial_c.expand(-1, batch_size, -1).contiguous()
                        initial_state = (initial_h, initial_c)
        
                    elif self.model.name.split('_')[0] == 'RNNModel':


                        initial_state = self.model.initial_hidden_state.expand(-1, batch_size, -1).contiguous()


                    if self.model.ext_vars is not None:


                        outputs_forward = self.model.forward(input_sequence=batch_inps[: ,:,:- self.model.ext_vars], 
                                                    initial_states=initial_state, 
                                                    batch_of_externals = batch_inps[:,:, - self.model.ext_vars:])#, externals=externals)

                        batch_outs = batch_outs[:,:,: - self.model.ext_vars]

                        loss = self.loss_fn(outputs_forward.output_sequence, batch_outs)
                    
                    else:

                        outputs_forward = self.model.forward(input_sequence=batch_inps, initial_states=initial_state)#, externals=externals)



                    loss = self.loss_fn(outputs_forward.output_sequence, batch_outs)
                
                    batch_count += 1

                    accumulated_loss += loss


                self.optimizer.zero_grad()
                accumulated_loss.backward()
                self.optimizer.step()


                


                self.model.eval() 

                if self.model.ext_vars is not None:


                    
                    with torch.no_grad():

                        out_seq, forecasts = self.model.forecast_seq_to_seq(
                            calibration_window=calibration_window[:,:-self.model.ext_vars] , 
                            n_steps=val_data.shape[0],
                            externals_for_calibration=calibration_window[:,-self.model.ext_vars:] ,
                            externals_for_forecasts=val_data[:,-self.model.ext_vars:])
                        
                else:
                    
                    with torch.no_grad():
                        out_seq, forecasts = self.model.forecast_seq_to_seq(
                            calibration_window=calibration_window, n_steps=val_data.shape[0])
                        



                current_val_loss , best_epoch_val_loss = self.save_best_model_based_on_val_loss(epoch=epoch, 
                                                                                    out_seq=out_seq,
                                                                                    forecasts=forecasts,
                                                                                    val_data=val_data,
                                                                                save_path="best_val_avg_loss.pth")
                            
                    
                epoch_losses.append({'epoch': epoch+1,'avg_train_loss': accumulated_loss.item()/batch_count,
                                          'val_loss':current_val_loss})



                print(f"[End of Epoch {epoch+1}] Current val loss: {current_val_loss:.6f} || Best val Loss: {best_epoch_val_loss:.6f}")

            self.save_epochs_losses_to_json(epoch_losses=epoch_losses, filename="epoch_losses.json")




    def save_forecasts_to_csv(self, out_seq: torch.Tensor, 
                                var_names: List[str],
                              forecast: torch.Tensor,
                                filename: str = 'forecast_results.csv') -> None:
        """
        Save all forecast outputs (per epoch) to a CSV file.

        Parameters
        ----------
        out_seq : torch.Tensor
            Output sequence from the model, shape: (n_steps, output_size)

        var_names : List[str]
            List of variable names for the columns in the CSV file.

        forecast : torch.Tensor
            Forecasted outputs, shape: (n_steps, output_size)

        filename : str
            Path to the CSV file to save forecasts.
        """
        all_data = []

        # for epoch_idx, (out_seq, forecast) in enumerate(self.forecast_results):
        #     if out_seq is None or forecast is None:
        #         continue

            # Detach and convert to numpy
        out_seq_np = out_seq.cpu().numpy()
        forecast_np = forecast.cpu().numpy()

        for t, val in enumerate(out_seq_np):
            all_data.append({
                # "epoch": epoch_idx + 1,
                "type": "context_output",
                "timestep": t,
                **{f"dim_{i}": v for i, v in enumerate(val)}
            })

        for t, val in enumerate(forecast_np):
            all_data.append({
                # "epoch": epoch_idx + 1,
                "type": "forecast",
                "timestep": t,
                **{f"dim_{i}": v for i, v in enumerate(val)}
            })

        df = pd.DataFrame(all_data)
        if var_names is not None:
            for i, var_name in enumerate(var_names):
                df.rename(columns={f"dim_{i}": var_name}, inplace=True)

        df.to_csv(os.path.join(self.save_dir, filename), index=False)
        print(f"Forecasts saved to: {filename}")

    def save_epochs_losses_to_json(self, epoch_losses: List[float], filename: str = 'epoch_losses.json') -> None:
        """
        Save the epoch losses to a JSON file.

        Parameters
        ----------
        epoch_losses : List[float]
            List of average loss per epoch.

        filename : str
            Path to the JSON file to save epoch losses.
        """
        with open(os.path.join(self.save_dir, filename), 'w') as f:
            json.dump(epoch_losses, f, indent=4)
        print(f"Epoch losses saved to: {filename}")




    def save_best_model_based_on_train_loss(self, epoch:int, train_loss : float, save_path: str = "best_train_avg_loss.pth") -> None:

        best_loss =  float('inf')
        """
        Save the model state_dict if the latest epoch has the best (lowest) average training loss.

        Parameters
        ----------
        save_path : str
            Path where the best model checkpoint will be saved.
        """
        if not train_loss:
            print("No training loss recorded.")
            return



        if train_loss < best_loss:

            self.model.save_checkpoint(epoch=epoch + 1, loss=train_loss, optimizer=self.optimizer, checkpoint_dir=self.save_dir, cleanup=True)
            
            
            print(f"✅ Best average train loss so far is ({train_loss:.6f}). Model saved to {save_path}")
            best_loss = train_loss
        else:
            print(f"ℹ️ Average loss {train_loss:.6f} not better than best {best_loss:.6f}. Skipping save.")

        return best_loss


    
    def save_best_model_based_on_val_loss(self, epoch:int, out_seq:torch.Tensor, 
                                            forecasts: torch.Tensor,  val_data: torch.Tensor,
                               save_path: str = "best_val_loss.pth") -> None:
        

        """
        Save the model state_dict if the current forecast has the best (lowest) MSE against ground truth.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        forecasts: Optional[torch.Tensor]
            Forecasted outputs, shape: (n_steps, output_size)

        val_data: Optional[torch.Tensor]
            Ground truth which is your validation sequence. Shape: (n_steps, [output_size + ext_vars] or (n_steps, [output_size])
        

        save_path : str
            Path where the best model checkpoint will be saved.
        """
        best_val_loss = float('inf')


        if self.model.ext_vars is not None:
            current_val_loss = self.loss_fn(forecasts, val_data[:,:-self.model.ext_vars]).item()
        
        else:
            current_val_loss = self.loss_fn(forecasts, val_data).item()


        if current_val_loss < best_val_loss:
            self.model.save_checkpoint(epoch=epoch + 1, loss=current_val_loss, optimizer=self.optimizer, checkpoint_dir=self.save_dir, cleanup=True)
            best_val_loss =  current_val_loss
            self.save_forecasts_to_csv(out_seq=out_seq, var_names=self.variable_names, forecast=forecasts, filename=f"forecast_results_epoch_{epoch+1}.csv")

            print(f"✅ New best forecast MSE: {current_val_loss:.6f}. Model saved to {save_path}")
        else:
            print(f"ℹ️ Forecast MSE {current_val_loss:.6f} not better than best {best_val_loss:.6f}. Skipping save.")
        return current_val_loss , best_val_loss
