import os , json, re, csv, torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
from typing import Optional, Literal
from ..model_utils.custom_losses import LogCoshLoss
from ..models.HCNN.hcnn_models import Vanilla_Model, PTF_Model, LForm_Model, LSpa_Model

class HCNNTrainer:
    def __init__(
        self,
        model: nn.Module , 
        loss_fn: Literal["mse", "logcosh"]   =  "mse" ,
        save_dir: Optional[str] = "./checkpoints_single",
        backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch' , 
        optimizer_type: Literal['adam', 'sgd'] = 'adam',
        learning_rate: float = 1e-4,
        variable_names: Optional[List[str]] = None):
        

        self.model = model
        # self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type
        self.backprop_mode =  backprop_mode
        self.variable_names = variable_names or [f"var_{i+1}" for i in range(self.model.n_obs_vars)]
        self.loss_fn = nn.MSELoss() if loss_fn == "mse" else LogCoshLoss()
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) if self.optimizer_type == "adam" \
                          else torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)


        self.variable_names = variable_names or [f"var_{i+1}" for i in range(self.model.n_obs_vars)]
        # self.device = ensemble.models[0]._get_default_device()

        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            
        self.save_dir = f"{save_dir}_{self.model.name.split('_')[0]}_{self.backprop_mode}_{timestamp}"


        os.makedirs(self.save_dir, exist_ok=True)
            



    def save_epochs_losses_to_json(self, epoch_losses: List[float], filename: str = 'ensemble_epoch_losses.json') -> None:
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



    def load_single_model_checkpoints(self, checkpoint_paths: str , train_type :str) -> List[dict]:
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

        if train_type == "train_only":

            if len(os.listdir(checkpoint_paths)) != 1:
                raise ValueError(f"Expected 1 checkpoint paths, got {len(os.listdir(checkpoint_paths))}")



            for idx, checkpoint_path in enumerate(sorted(os.listdir(checkpoint_paths))):
                checkpoint_path_ =  os.path.join(self.save_dir, f"{checkpoint_path}")
                if not os.path.isfile(checkpoint_path_):
                    raise FileNotFoundError(f"❌ Checkpoint not found at: {checkpoint_path_}")
                

                # Load checkpoint
                checkpoint = torch.load(checkpoint_path_, map_location=self.model.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                # self.model.load_state_dict(checkpoint["optimizer_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 

                print(f"✅ Loaded {self.model.name.split("_")[0]}  from {checkpoint_path_} | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.6f}")

        elif train_type == "train_and_validate":

            if len(os.listdir(checkpoint_paths)) != 2:
                raise ValueError(f"Expected CSV File in there. Please check it again!")

            # load checkpoint paths excludig csv files
            for idx, checkpoint_path in enumerate(sorted(os.listdir(checkpoint_paths))):

                if checkpoint_path.lower().endswith('.csv'):
                    continue
                
                checkpoint_path_ =  os.path.join(self.save_dir, f"{checkpoint_path}")
                if not os.path.isfile(checkpoint_path_):
                    raise FileNotFoundError(f"❌ Checkpoint not found at: {checkpoint_path_}")
                


                # Load checkpoint
                checkpoint = torch.load(checkpoint_path_, map_location=self.model.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                # self.model.load_state_dict(checkpoint["optimizer_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 


                print(f"✅ Loaded {self.model.name.split("_")[0]} from {checkpoint_path_} | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.6f}")

            

        return self.model , self.optimizer

    def train_only(self, data_loader , num_epochs: int = 10, verbose: bool = True):

        
        
        best_median_loss = float("inf")


        best_loss = float('inf')
        best_epoch = -1
        train_type = 'train_only'
        
        # if self.best_on == 'individual':


        if self.backprop_mode == 'per_batch':

            # epoch_losses = []


            epoch_loss_summary = []
            member_loss_summary = {}
            for epoch in tqdm(range(num_epochs), desc="Training Only"):


                

                
                # for idx , model in enumerate(self.ensembles):
                batch_loss_summary =  []
                batch_losses = []

                self.model.train()
                total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                batch_idx = 0


                for batch in data_loader:
                    batch_count+=1
                    self.optimizer.zero_grad()
                    results = self.model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()

                    self.optimizer.step()

                    batch_loss_summary.append({f'batch_idx_{batch_count}': loss.item()})

                    batch_losses.append(loss.item())
                    total_loss += loss.item()
                    if loss.item() < best_loss:
                        best_epoch = epoch + 1
                        self.model.save_checkpoint(epoch=best_epoch, loss=loss.item(), optimizer=self.optimizer, checkpoint_dir=self.save_dir ,  
                                                   add_stuffs=f"_single",cleanup=True)
                        best_loss = loss.item()
                        
                        print(f"✅  New best batch train loss: {loss:.6f} | Current batch train loss: {loss.item():.6f} |")
                    else:
                        print(f"ℹ️ [Current batch Train loss {loss:.6f} not better than best batch train loss{best_loss:.6f}. Skipping save.")


                # avg_train_loss = total_loss / batch_count
                
                # member_loss_summary[f'member_{idx+1}'] = 
                epoch_loss_summary.append({f'epoch_{epoch+1}' :{'avg_loss':float(np.mean(batch_losses)), 'median_loss':float(np.median(batch_losses))}})

                print(f" End of Epoch {epoch+1} | Best Epoch {best_epoch}| All batches processed  for {self.model.name } | Current batch train loss: {loss.item():.6f}|  Best batch train Loss: {best_loss:.6f}")
                print("\n")
            
            



                self.model , self.optimizer =  self.load_single_model_checkpoints(self.save_dir, train_type)

            self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)



        elif self.backprop_mode == 'per_epoch':

            epoch_losses = []

            epoch_loss_summary = []
            member_loss_summary ={}

            for epoch in tqdm(range(num_epochs), desc="Training Only"):

                batch_idx = 0
                

                best_loss = float('inf')

                # for idx , model in enumerate(self.ensembles):

                    # optimizer = model[f'optimizer_{idx+1}']
                self.model.train()
                total_loss , accumulated_loss, batch_count = 0.0, 0.0, 0
                batches_loss = []
                for batch in data_loader:
                    batch_count+=1
                    # optimizer.zero_grad()
                    results = self.model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    accumulated_loss += loss

                    batches_loss.append(loss.item())
                total_loss = accumulated_loss / batch_count

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_loss_summary.append({f'epoch_{epoch+1}' :{'avg_loss':float(np.mean(batches_loss)), 'median_loss':float(np.median(batches_loss))}})

                
                


                if total_loss.item() < best_loss:
                    best_epoch = epoch + 1
                    self.model.save_checkpoint(epoch=best_epoch, loss=total_loss.item(), optimizer=self.optimizer, checkpoint_dir=self.save_dir , 
                                               add_stuffs=f"_single", cleanup=True)
                    best_loss = total_loss.item()
                    
                    print(f"✅ New best epoch train loss: {best_loss:.6f} |  Current epoch train loss: {total_loss.item():.6f} |")

                else:
                    print(f"ℹ️ Current epoch Train loss {total_loss.item():.6f} not better than best epoch train loss {best_loss:.6f}. Skipping save.")




                print(f" End of Epoch {epoch+1} | Best Epoch {best_epoch}| All batches processed  for {self.model.name } | Current  epoch train loss: {loss.item():.6f}|  Best epoch train Loss: {best_loss:.6f}")
                print("\n")

                self.model , self.optimizer =  self.load_single_model_checkpoints(self.save_dir, train_type)

                    

            self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)

        


    def train_and_validate(
        self,
        data_loader,
        num_epochs: int, 
        calibration_window: torch.Tensor,
        val_data: torch.Tensor,
        verbose: bool = True ):

        
        
        best_median_loss = float("inf")


        # best_loss = float('inf')
        best_epoch = -1
        train_type = 'train_and_validate'
        
        # if self.best_on == 'individual':


        if self.backprop_mode == 'per_batch':

            # epoch_losses = []


            epoch_loss_summary = []
            member_loss_summary = {}
            best_loss = float('inf')

            for epoch in tqdm(range(num_epochs), desc="Training and validating"):  
                # for idx , model in enumerate(self.ensembles):
                batch_train_loss_summary =  []
                batch_val_loss_summary = []
                batch_losses = []
                batch_val_loss=[]

                # optimizer = model[f'optimizer_{idx+1}']
                self.model.train()
                total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                batch_idx = 0


                for batch in data_loader:
                    batch_count+=1
                    self.optimizer.zero_grad()
                    results = self.model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    loss.backward()

                    self.optimizer.step()

                    batch_train_loss_summary.append({f'batch_idx_{batch_count}': loss.item()})

                    batch_losses.append(loss.item())
                    total_loss += loss.item()


                    self.model.eval()

                    with torch.no_grad():
                        results = self.model.forward(
                            data_window=calibration_window.unsqueeze(0),
                            forecast_horizon=val_data.shape[0],
                        )
                        forecast = results.forecasts.squeeze(0)
                        val_loss = self.loss_fn(forecast, val_data).item()
                        batch_val_loss_summary.append({f'batch_idx_{batch_count}': val_loss})
                        batch_val_loss.append(val_loss)
                        # epoch_val_losses.append(val_loss)
                    if val_loss < best_loss:

                        best_epoch = epoch + 1
                        self. model.save_checkpoint(epoch=best_epoch, loss=loss.item(),
                                                                    optimizer=self.optimizer, checkpoint_dir=self.save_dir , 
                                                                    cleanup=True,add_stuffs=f"_single")
                        best_loss = val_loss
                        

                        self.save_forecasts_to_csv(out_seq=results.expectations.squeeze(0) ,
                                                    var_names=[f"var_{i+1}" for i in range(self.model.n_obs_vars)],
                                                    forecast=forecast,filename=f"forecasts_epoch_{epoch+1}_single_.csv")
                                                    

                        print(f"✅ New best val loss: {best_loss:.6f} | Current val loss: {val_loss}" )
                    else:
                        print(f"ℹ️ Current val loss {val_loss:.6f} not better than best val loss {best_loss:.6f}. Skipping save.")


                # avg_train_loss = total_loss / batch_count
                
                epoch_loss_summary.append({f'epoch_{epoch+1}' :{'avg_train_loss':float(np.mean(batch_losses)), 'median_train_loss':float(np.median(batch_losses)),
                                                            'avg_val_loss': float(np.mean(batch_val_loss)), 'median_val_loss': float(np.median(batch_val_loss))}})

                print(f" End of Epoch {epoch+1} | Best Epoch {best_epoch}| All batches processed  for {self.model.name } | Current val_loss: {val_loss:.6f}|  Best val Loss: {best_loss:.6f}")
                print("\n")

            



                self.model , self.optimizer =  self.load_single_model_checkpoints(self.save_dir, train_type)

            self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)


        elif self.backprop_mode == 'per_epoch':

            epoch_losses = []
            batch_val_loss_summary = []
            epoch_loss_summary = []
            member_loss_summary ={}

            for epoch in tqdm(range(num_epochs), desc="Training and validating"):  

                batch_idx = 0
                

                best_loss = float('inf')

                # for idx , model in enumerate(self.ensembles):

                    # optimizer = model[f'optimizer_{idx+1}']
                self.model.train()
                total_loss , accumulated_loss, batch_count = 0.0, 0.0, 0
                batches_loss = []
                for batch in data_loader:
                    batch_count+=1
                    # optimizer.zero_grad()
                    results = self.model.forward(data_window=batch)
                    loss = self.loss_fn(results.expectations, batch)
                    accumulated_loss += loss

                    batches_loss.append(loss.item())
                total_loss = accumulated_loss / batch_count

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()


                # batches_loss.append(total_loss.item())

                # avg_calc_nested = AverageCalculator(member_loss_summary)
                print(f"[Epoch {epoch+1} Processed all batches for Model {self.model.name}] Average Train Loss: {float(np.mean(batches_loss)):.6f}")
                
                self.model.eval()

                with torch.no_grad():
                    results = self.model.forward(
                        data_window=calibration_window.unsqueeze(0),
                        forecast_horizon=val_data.shape[0],
                    )
                    forecast = results.forecasts.squeeze(0)
                    val_loss = self.loss_fn(forecast, val_data).item()
                    batch_val_loss_summary.append(val_loss)


                if val_loss < best_loss:
                    best_epoch = epoch + 1
                    self.model.save_checkpoint(epoch=best_epoch , loss=loss.item(),
                                                                optimizer=self.optimizer, checkpoint_dir=self.save_dir , 
                                                                cleanup=True,add_stuffs=f"_single")
                    best_loss = val_loss
                    # best_epoch = epoch + 1

                    self.save_forecasts_to_csv(out_seq=results.expectations.squeeze(0) ,
                                                var_names=[f"var_{i+1}" for i in range(self.model.n_obs_vars)],
                                                forecast=forecast,filename=f"forecasts_epoch_{epoch+1}_single_.csv")


                    print(f"✅ New best val loss: {best_loss:.6f} | Current val loss: {val_loss}" )
                else:
                    print(f"ℹ️ Current val loss {val_loss:.6f} not better than best val loss {best_loss:.6f}. Skipping save.")



                epoch_loss_summary.append({f'epoch_{epoch+1}' : {'avg_train_loss':float(np.mean(batches_loss)),
                                                                    'median_train_loss':float(np.median(batches_loss)),
                                                                    'val_loss': val_loss}})
                # print(f"[End of Epoch {epoch+1}  =================================================================")
                print(f" End of Epoch {epoch+1} | Best Epoch {best_epoch}| All batches processed  for {self.model.name } | Current val_loss: {val_loss:.6f}|  Best val Loss: {best_loss:.6f}")
                print("\n")

                self.model , self.optimizer =  self.load_single_model_checkpoints(self.save_dir, train_type)
                

            self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)


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