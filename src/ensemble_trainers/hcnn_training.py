import pandas as pd
import torch , os, csv
import matplotlib.pyplot as plt
# from .utils import  custom_fcts
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Literal
from glob import glob
import json, os
# from .models import Vanilla_Model , LSpa_Model , LForm_Model, PTF_Model
from datetime import datetime

class HCNNEnsembleTrainer:
    def __init__(
        self,
        ensembles: List,
        loss_fn: Literal["mse", "logcosh"]   =  "mse" ,
        save_dir: Optional[str] = "./checkpoints_ensemble",
        best_on: Literal["median", "individual"] = "individual",
        variable_names: Optional[List[str]] = None):
        
        self.variable_names = variable_names or [f"var_{i+1}" for i in range(ensembles[0][f"member_1"].n_obs_vars)]
        self.ensembles = ensembles
        # self.optimizer = optimizer

        if loss_fn.lower() == "mse":
            self.loss_fn = nn.MSELoss()

        elif loss_fn.lower() == "logcosh":
            self.loss_fn = LogCoshLoss()
        self.best_on = best_on
        self.variable_names = variable_names or [f"var_{i+1}" for i in range(ensembles[0][f"member_1"].n_obs_vars)]
        # self.device = ensemble.models[0]._get_default_device()

        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            
        # self.loss_fn = nn.MSELoss() if loss_fn == "mse" else nn.LogCoshLoss()
        self.save_dir = f"{save_dir}_{self.ensembles[0]['member_1'].name.split(':')[1].split('_')[0] }_{self.best_on}_{timestamp}"
        # self.models =  []
        # self.loss_histories: List[List[float]] = []
        # self.best_losses: List[float] = [float('inf')] * n_ensemble
        # self.best_on = best_on

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



    def load_ensemble_checkpoints(self, checkpoint_paths: str , train_type :str) -> List[dict]:
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

            if len(os.listdir(checkpoint_paths)) != len(self.ensembles):
                raise ValueError(f"Expected {len(self.ensembles)} checkpoint paths, got {len(os.listdir(checkpoint_paths))}")



            for idx, checkpoint_path in enumerate(sorted(os.listdir(checkpoint_paths))):
                checkpoint_path_ =  os.path.join(self.save_dir, f"{checkpoint_path}")
                if not os.path.isfile(checkpoint_path_):
                    raise FileNotFoundError(f"❌ Checkpoint not found at: {checkpoint_path_}")
                

                # Load checkpoint
                checkpoint = torch.load(checkpoint_path_, map_location=self.ensembles[idx][f'member_{idx+1}'].device)
                self.ensembles[idx][f'member_{idx+1}'].load_state_dict(checkpoint["model_state_dict"])
                self.ensembles[idx][f'optimizer_{idx+1}'].load_state_dict(checkpoint["optimizer_state_dict"])

                print(f"✅ Loaded Model {idx+1} from {checkpoint_path_} | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.6f}")

        elif train_type == "train_and_validate":

            if len(os.listdir(checkpoint_paths)) != len(self.ensembles)+len(self.ensembles):
                raise ValueError(f"Expected CSV File in there. Please check it again!")

            # load checkpoint paths excludig csv files
            for idx, checkpoint_path in enumerate(sorted(os.listdir(checkpoint_paths))):

                if checkpoint_path.lower().endswith('.csv'):
                    continue
                
                checkpoint_path_ =  os.path.join(self.save_dir, f"{checkpoint_path}")
                if not os.path.isfile(checkpoint_path_):
                    raise FileNotFoundError(f"❌ Checkpoint not found at: {checkpoint_path_}")
                


                # Load checkpoint
                checkpoint = torch.load(checkpoint_path_, map_location=self.ensembles[idx][f'member_{idx+1}'].device)
                self.ensembles[idx][f'member_{idx+1}'].load_state_dict(checkpoint["model_state_dict"])
                self.ensembles[idx][f'optimizer_{idx+1}'].load_state_dict(checkpoint["optimizer_state_dict"])

                print(f"✅ Loaded Model {idx+1} from {checkpoint_path_} | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.6f}")

            

        return self.ensembles

    def train_only(
        self,
        data_loader,
        num_epochs: int = 10,
        verbose: bool = True, backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch'):

        
        
        best_median_loss = float("inf")


        best_losses = [float('inf')] * len(self.ensembles)
        best_epoch = -1
        train_type = 'train_only'
        
        if self.best_on == 'individual':


            if backprop_mode == 'per_batch':

                # epoch_losses = []


                epoch_loss_summary = []

                for epoch in tqdm(range(num_epochs), desc="Training Only"):  


                    

                    member_loss_summary = {}
                    for idx , model in enumerate(self.ensembles):
                        batch_loss_summary =  []
                        batch_losses = []

                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                        batch_idx = 0


                        for batch in data_loader:
                            batch_count+=1
                            optimizer.zero_grad()
                            results = model[f'member_{idx+1}'].forward(data_window=batch)
                            loss = self.loss_fn(results.expectations, batch)
                            loss.backward()

                            optimizer.step()

                            batch_loss_summary.append({f'batch_idx_{batch_count}': loss.item()})

                            batch_losses.append(loss.item())
                            total_loss += loss.item()
                            if loss.item() < best_losses[idx]:
                                model[f'member_{idx+1}'].save_checkpoint(epoch=epoch + 1, loss=loss.item(), optimizer=optimizer, checkpoint_dir=self.save_dir ,  cleanup=True)
                                best_losses[idx] = loss.item()
                                best_epoch = epoch + 1
                                print(f"✅ [Model {idx+1}] New best train loss: {loss:.6f}")
                            else:
                                print(f"ℹ️ [Model {idx+1}] Train loss {loss:.6f} not better than best {best_losses[idx]:.6f}. Skipping save.")


                        # avg_train_loss = total_loss / batch_count
                        
                        member_loss_summary[f'member_{idx+1}'] = {'avg_loss':float(np.mean(batch_losses)), 'median_loss':float(np.median(batch_losses))}

                    print(f"[Epoch {epoch+1} Processed all batches for Model member {idx+1}] Current batch_loss: {loss.item():.6f}|  Best batch Loss: {best_losses[idx]:.6f}")
                    
                    epoch_loss_summary.append({f'epoch_{epoch+1}' :member_loss_summary})
                    # print(f"Epoch {epoch+1} | Best Median Loss: {float(np.median(batch_losses)):.6f} | Best Average Loss: {float(np.mean(batch_losses)):.6f}")
                
                    print(f"[End of Epoch {epoch+1}  =================================================================")


                    self.ensembles =  self.load_ensemble_checkpoints(self.save_dir, train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)


            elif backprop_mode == 'per_epoch':

                epoch_losses = []

                epoch_loss_summary = []

                for epoch in tqdm(range(num_epochs), desc="Training Only"):  

                    batch_idx = 0
                    member_loss_summary ={}

                    best_losses = [float('inf')] * len(self.ensembles)

                    for idx , model in enumerate(self.ensembles):

                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss , accumulated_loss, batch_count = 0.0, 0.0, 0
                        batches_loss = []
                        for batch in data_loader:
                            batch_count+=1
                            # optimizer.zero_grad()
                            results = model[f'member_{idx+1}'].forward(data_window=batch)
                            loss = self.loss_fn(results.expectations, batch)
                            accumulated_loss += loss

                            batches_loss.append(loss.item())
                        total_loss = accumulated_loss / batch_count

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        member_loss_summary[f'member_{idx+1}'] = {'avg_loss':float(np.mean(batches_loss)), 'median_loss':float(np.median(batches_loss))}

                        # batches_loss.append(total_loss.item())

                        # avg_calc_nested = AverageCalculator(member_loss_summary)
                        print(f"[Epoch {epoch+1} Processed all batches for Model member {idx+1}] Average Train Loss: {float(np.mean(batches_loss)):.6f}")
                        


                        if total_loss.item() < best_losses[idx]:
                            model[f'member_{idx+1}'].save_checkpoint(epoch=epoch + 1, loss=total_loss.item(), optimizer=optimizer, checkpoint_dir=self.save_dir ,  cleanup=True)
                            best_losses[idx] = total_loss.item()
                            best_epoch = epoch + 1
                            print(f"✅ [Model {idx+1}] New best train loss: {total_loss.item():.6f}")

                        else:
                            print(f"ℹ️ [Model {idx+1}] Train loss {total_loss.item():.6f} not better than best {best_losses[idx]:.6f}. Skipping save.")


                    # epoch_losses.append({f'epoch_{epoch+1}': member_loss_summary})
                    epoch_loss_summary.append({f'epoch_{epoch+1}' : member_loss_summary})
                    print(f"[End of Epoch {epoch+1}  =================================================================")

                    self.ensembles =  self.load_ensemble_checkpoints(self.save_dir, train_type)
                        

                self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)

        
        elif self.best_on == 'median':


            if backprop_mode == 'per_batch':

                epoch_losses = []



                epoch_loss_summary = []
                for epoch in tqdm(range(num_epochs), desc="Training Only"):  
                    
                    best_losses = [float('inf')] * len(self.ensembles)

                    batch_losses = []

                    batch_idx = 0
                    batch_loss_summary =  {}
                    for batch in data_loader:
                        batch_idx+=1
                        member_loss_summary = {}
                        for idx , model in enumerate(self.ensembles):
                            optimizer = model[f'optimizer_{idx+1}']
                            model[f'member_{idx+1}'].train()
                            total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                            batch_count = 0

                            optimizer.zero_grad()
                            results = model[f'member_{idx+1}'].forward(data_window=batch)
                            loss = self.loss_fn(results.expectations, batch)
                            loss.backward()

                            optimizer.step()

                            member_loss_summary[f'member_{idx+1}'] = {'train_loss': loss.item()}
                            batch_losses.append(loss.item())

                        
                        if float(np.median(batch_losses)) < float(np.median(best_losses)):
                            for idx , model in enumerate(self.ensembles):
                                model[f'member_{idx+1}'].save_checkpoint(epoch=epoch + 1, loss=float(np.median(batch_losses)), optimizer=optimizer, checkpoint_dir=self.save_dir ,  cleanup=True)
                                best_losses[idx]  =  batch_losses[idx]

                            print(f"✅ All ensemble members saved based on median loss: {float(np.median(batch_losses)):.6f}")

                        else:
                            print(f"ℹ️ All ensemble members not saved based as median loss: {float(np.median(batch_losses)):.6f} not better than best {float(np.median(best_losses)):.6f}. Skipping save.")

                        batch_loss_summary[f'batch_idx_{batch_idx}'] = member_loss_summary   

                    
                        print(f"[Epoch {epoch+1} Processed All  Models through batch {batch_idx} |  Best Median Loss: {float(np.median(best_losses)):.6f}")

                    epoch_loss_summary.append({f'epoch_{epoch+1}' : {'avg_loss': float(np.median(batch_losses)), 'median_loss': float(np.median(best_losses))}})

                    print(f"[End of Epoch {epoch+1}  =================================================================")

                    self.ensembles =  self.load_ensemble_checkpoints(self.save_dir, train_type)
                    
                self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)





            elif backprop_mode == 'per_epoch':

                epoch_losses = []

                epoch_loss_summary = []


                for epoch in tqdm(range(num_epochs), desc="Training Only"):  

                    batch_idx = 0
                    member_loss_summary ={}
                    member_loss = []

                    best_losses = [float('inf')] * len(self.ensembles)
                    for idx , model in enumerate(self.ensembles):

                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss , accumulated_loss, batch_count = 0.0, 0.0, 0
                        
                        for batch in data_loader:
                            batch_count+=1
                            # optimizer.zero_grad()
                            results = model[f'member_{idx+1}'].forward(data_window=batch)
                            loss = self.loss_fn(results.expectations, batch)
                            accumulated_loss += loss


                        total_loss = accumulated_loss / batch_count

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        member_loss_summary[f'member_{idx+1}'] = {'train_loss': total_loss.item()}
                        member_loss.append(total_loss.item())
                        # avg_calc_nested = AverageCalculator(member_loss_summary)

                        print(f"[Epoch {epoch+1} Processed all batches for Model member {idx+1}] Average Train Loss: {total_loss.item():.6f}")


                    if float(np.median(member_loss)) < float(np.median(best_losses)):
                        for idx , model in enumerate(self.ensembles):
                            model[f'member_{idx+1}'].save_checkpoint(epoch=epoch + 1, loss=float(np.median(member_loss)), optimizer=optimizer, checkpoint_dir=self.save_dir ,  cleanup=True)
                            best_losses[idx]  =  member_loss[idx]

                        print(f"✅ All ensemble members saved based on median loss: {float(np.median(member_loss)):.6f}")

                    else:
                        print(f"ℹ️ All ensemble members not saved based as median loss: {float(np.median(member_loss)):.6f} not better than best {float(np.median(best_losses)):.6f}. Skipping save.")

                    epoch_loss_summary.append({f'epoch_{epoch+1}' : member_loss_summary})

                    print(f"[End of Epoch {epoch+1}  =================================================================")

                    self.ensembles =  self.load_ensemble_checkpoints(self.save_dir, train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)




    def train_and_validate(
        self,
        data_loader,
        num_epochs: int, 
        calibration_window: torch.Tensor,
        val_data: torch.Tensor,
        verbose: bool = True, backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch'):

        
        
        best_median_loss = float("inf")


        best_losses = [float('inf')] * len(self.ensembles)
        best_epoch = -1
        train_type = 'train_and_validate'
        
        if self.best_on == 'individual':


            if backprop_mode == 'per_batch':

                # epoch_losses = []


                epoch_loss_summary = []

                for epoch in tqdm(range(num_epochs), desc="Training and validating"):  


                    

                    member_loss_summary = {}
                    for idx , model in enumerate(self.ensembles):
                        batch_train_loss_summary =  []
                        batch_val_loss_summary = []
                        batch_losses = []
                        batch_val_loss=[]

                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                        batch_idx = 0


                        for batch in data_loader:
                            batch_count+=1
                            optimizer.zero_grad()
                            results = model[f'member_{idx+1}'].forward(data_window=batch)
                            loss = self.loss_fn(results.expectations, batch)
                            loss.backward()

                            optimizer.step()

                            batch_train_loss_summary.append({f'batch_idx_{batch_count}': loss.item()})

                            batch_losses.append(loss.item())
                            total_loss += loss.item()


                            model[f'member_{idx+1}'].eval()

                            with torch.no_grad():
                                results = model[f'member_{idx+1}'].forward(
                                    data_window=calibration_window.unsqueeze(0),
                                    forecast_horizon=val_data.shape[0],
                                )
                                forecast = results.forecasts.squeeze(0)
                                val_loss = self.loss_fn(forecast, val_data).item()
                                batch_val_loss_summary.append({f'batch_idx_{batch_count}': val_loss})
                                batch_val_loss.append(val_loss)
                                # epoch_val_losses.append(val_loss)
                            if val_loss < best_losses[idx]:
                                model[f'member_{idx+1}'].save_checkpoint(epoch=epoch + 1, loss=loss.item(),
                                                                          optimizer=optimizer, checkpoint_dir=self.save_dir , 
                                                                            cleanup=True,add_stuffs=f"_member_{idx+1}")
                                best_losses[idx] = val_loss
                                best_epoch = epoch + 1

                                self.save_forecasts_to_csv(out_seq=results.expectations.squeeze(0) ,
                                                           var_names=[f"var_{i+1}" for i in range(model[f'member_{idx+1}'].n_obs_vars)],
                                                           forecast=forecast,filename=f"forecasts_epoch_{epoch+1}_member_{idx+1}_.csv")
                                                           

                                print(f"✅ [Model Member {idx+1}] New best val loss: {val_loss:.6f}")
                            else:
                                print(f"ℹ️ [Model Member {idx+1}] Val loss {val_loss:.6f} not better than best {best_losses[idx]:.6f}. Skipping save.")


                        # avg_train_loss = total_loss / batch_count
                        
                        member_loss_summary[f'member_{idx+1}'] = {'avg_train_loss':float(np.mean(batch_losses)), 'median_train_loss':float(np.median(batch_losses)),
                                                                    'avg_val_loss': float(np.mean(batch_val_loss)), 'median_val_loss': float(np.median(batch_val_loss))}

                    print(f"[Epoch {epoch+1} Processed all batches for Model member {idx+1}] Current val_loss: {val_loss:.6f}|  Best val Loss: {best_losses[idx]:.6f}")
                    
                    epoch_loss_summary.append({f'epoch_{epoch+1}' :member_loss_summary})
                    # print(f"Epoch {epoch+1} | Best Median Loss: {float(np.median(batch_losses)):.6f} | Best Average Loss: {float(np.mean(batch_losses)):.6f}")
                
                    print(f"[End of Epoch {epoch+1}  =================================================================")


                    self.ensembles =  self.load_ensemble_checkpoints(self.save_dir, train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)


            elif backprop_mode == 'per_epoch':

                epoch_losses = []

                epoch_loss_summary = []

                for epoch in tqdm(range(num_epochs), desc="Training and validating"):  

                    batch_idx = 0
                    member_loss_summary ={}

                    best_losses = [float('inf')] * len(self.ensembles)

                    for idx , model in enumerate(self.ensembles):

                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss , accumulated_loss, batch_count = 0.0, 0.0, 0
                        batches_loss = []
                        for batch in data_loader:
                            batch_count+=1
                            # optimizer.zero_grad()
                            results = model[f'member_{idx+1}'].forward(data_window=batch)
                            loss = self.loss_fn(results.expectations, batch)
                            accumulated_loss += loss

                            batches_loss.append(loss.item())
                        total_loss = accumulated_loss / batch_count

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()


                        # batches_loss.append(total_loss.item())

                        # avg_calc_nested = AverageCalculator(member_loss_summary)
                        print(f"[Epoch {epoch+1} Processed all batches for Model member {idx+1}] Average Train Loss: {float(np.mean(batches_loss)):.6f}")
                        
                        model[f'member_{idx+1}'].eval()

                        with torch.no_grad():
                            results = model[f'member_{idx+1}'].forward(
                                data_window=calibration_window.unsqueeze(0),
                                forecast_horizon=val_data.shape[0],
                            )
                            forecast = results.forecasts.squeeze(0)
                            val_loss = self.loss_fn(forecast, val_data).item()
                            batch_val_loss_summary.append(val_loss)
                            # epoch_val_losses.append(val_loss)
                        
                            member_loss_summary[f'member_{idx+1}'] = {'avg_train_loss':float(np.mean(batches_loss)),
                                                                       'median_train_loss':float(np.median(batches_loss)),
                                                                       'val_loss': val_loss}

                        if val_loss < best_losses[idx]:
                            model[f'member_{idx+1}'].save_checkpoint(epoch=epoch + 1, loss=loss.item(),
                                                                        optimizer=optimizer, checkpoint_dir=self.save_dir , 
                                                                        cleanup=True,add_stuffs=f"_member_{idx+1}")
                            best_losses[idx] = val_loss
                            best_epoch = epoch + 1

                            self.save_forecasts_to_csv(out_seq=results.expectations.squeeze(0) ,
                                                        var_names=[f"var_{i+1}" for i in range(model[f'member_{idx+1}'].n_obs_vars)],
                                                        forecast=forecast,filename=f"forecasts_epoch_{epoch+1}_member_{idx+1}_.csv")

                            print(f"✅ [Model Member {idx+1}] New best val loss: {val_loss:.6f}")
                        else:
                            print(f"ℹ️ [Model Member {idx+1}] Val loss {val_loss:.6f} not better than best {best_losses[idx]:.6f}. Skipping save.")



                    epoch_loss_summary.append({f'epoch_{epoch+1}' : member_loss_summary})
                    print(f"[End of Epoch {epoch+1}  =================================================================")

                    self.ensembles =  self.load_ensemble_checkpoints(self.save_dir, train_type)
                        

                self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)

        
        elif self.best_on == 'median':


            if backprop_mode == 'per_batch':

                epoch_losses = []



                epoch_loss_summary = []
                for epoch in tqdm(range(num_epochs), desc="Training and validating"):  
                    
                    best_losses = [float('inf')] * len(self.ensembles)
                    

                    batch_losses = []

                    batch_idx = 0
                    batch_val_loss_summary =  []
                    batch_loss_summary =  {}
                    for batch in data_loader:
                        batch_idx+=1
                        member_loss_summary = {}
                        forecasts_dict = {}
                        for idx , model in enumerate(self.ensembles):
                            optimizer = model[f'optimizer_{idx+1}']
                            model[f'member_{idx+1}'].train()
                            total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                            batch_count = 0

                            optimizer.zero_grad()
                            results = model[f'member_{idx+1}'].forward(data_window=batch)
                            loss = self.loss_fn(results.expectations, batch)
                            loss.backward()

                            optimizer.step()

                            # member_loss_summary[f'member_{idx+1}'] = {'batch_train_loss': loss.item()}
                            batch_losses.append(loss.item())

                            with torch.no_grad():
                                results = model[f'member_{idx+1}'].forward(
                                    data_window=calibration_window.unsqueeze(0),
                                    forecast_horizon=val_data.shape[0],
                                )
                                forecast = results.forecasts.squeeze(0)
                                val_loss = self.loss_fn(forecast, val_data).item()
                                batch_val_loss_summary.append(val_loss)

                                forecasts_dict[f'member_{idx+1}'] = forecast
                                # epoch_val_losses.append(val_loss)
                            
                                # member_loss_summary[f'member_{idx+1}'] = {'avg_train_loss':float(np.mean(batches_loss)),
                                #                                         'median_train_loss':float(np.median(batches_loss)),
                                #                                         'val_loss': val_loss}
                            member_loss_summary[f'member_{idx+1}'] = {'batch_train_loss': loss.item(),
                                                                      'val_loss': val_loss}


                        if float(np.median(batch_val_loss_summary)) < float(np.median(best_losses)):
                            for idx , model in enumerate(self.ensembles):
                                model[f'member_{idx+1}'].save_checkpoint(epoch=epoch + 1, loss=float(np.median(batch_losses)), 
                                                                         optimizer=optimizer, checkpoint_dir=self.save_dir ,  cleanup=True,
                                                                         add_stuffs=f"_member_{idx+1}")
                                best_losses[idx]  =  batch_losses[idx]


                                best_losses[idx] = val_loss
                                best_epoch = epoch + 1

                                self.save_forecasts_to_csv(out_seq=results.expectations.squeeze(0) ,
                                                            var_names=[f"var_{i+1}" for i in range(model[f'member_{idx+1}'].n_obs_vars)],
                                                            forecast=forecasts_dict[f'member_{idx+1}'],filename=f"forecasts_epoch_{epoch+1}_member_{idx+1}_.csv")

                            print(f"✅ All ensemble members saved based on median val loss: {float(np.median(batch_val_loss_summary)):.6f}")

                        else:
                            print(f"ℹ️ All ensemble members not saved based as median val loss: {float(np.median(batch_val_loss_summary)):.6f} not better than best {float(np.median(best_losses)):.6f}. Skipping save.")

                            
                        batch_loss_summary[f'batch_idx_{batch_idx}'] = member_loss_summary   

                    
                        print(f"[Epoch {epoch+1} Processed All  Models through batch {batch_idx} |  Best Median Loss: {float(np.median(best_losses)):.6f}")

                    epoch_loss_summary.append({f'epoch_{epoch+1}' : {'avg_train_loss': float(np.median(batch_losses)),
                                                                      'median_train_loss': float(np.median(batch_losses)),
                                                                      'avg_val_loss': float(np.mean(batch_val_loss_summary)),
                                                                      'median_val_loss': float(np.median(batch_val_loss_summary))}})

                    print(f"[End of Epoch {epoch+1}  =================================================================")

                    self.ensembles =  self.load_ensemble_checkpoints(self.save_dir, train_type)
                    
                self.save_epochs_losses_to_json(epoch_losses=epoch_loss_summary)





            elif backprop_mode == 'per_epoch':

                

                epoch_loss_summary = []

                batch_val_loss_summary = []
                for epoch in tqdm(range(num_epochs), desc="Training and validating"):  

                    batch_idx = 0
                    member_loss_summary ={}
                    member_loss = []
                    forecasts_dict = {}
                    best_losses = [float('inf')] * len(self.ensembles)
                    for idx , model in enumerate(self.ensembles):

                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss , accumulated_loss, batch_count = 0.0, 0.0, 0
                        
                        for batch in data_loader:
                            batch_count+=1
                            # optimizer.zero_grad()
                            results = model[f'member_{idx+1}'].forward(data_window=batch)
                            loss = self.loss_fn(results.expectations, batch)
                            accumulated_loss += loss


                        total_loss = accumulated_loss / batch_count

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        # batch_losses.append(total_loss.item())
                        member_loss_summary[f'member_{idx+1}'] = {'train_loss': total_loss.item()}
                        member_loss.append(total_loss.item())
                        # avg_calc_nested = AverageCalculator(member_loss_summary)

                        print(f"[Epoch {epoch+1} Processed all batches for Model member {idx+1}] Average Train Loss: {total_loss.item():.6f}")


                        with torch.no_grad():
                                    results = model[f'member_{idx+1}'].forward(
                                        data_window=calibration_window.unsqueeze(0),
                                        forecast_horizon=val_data.shape[0],
                                    )
                                    forecast = results.forecasts.squeeze(0)
                                    val_loss = self.loss_fn(forecast, val_data).item()
                                    batch_val_loss_summary.append(val_loss)

                                    member_loss_summary[f'member_{idx+1}'] = {'train_loss': total_loss.item(),
                                                                              'val_loss': val_loss}
                                    forecasts_dict[f'member_{idx+1}'] = forecast
                        
                    if float(np.median(batch_val_loss_summary)) < float(np.median(best_losses)):
                        for idx , model in enumerate(self.ensembles):
                            model[f'member_{idx+1}'].save_checkpoint(epoch=epoch + 1, 
                                                                        loss=float(np.median(member_loss)),
                                                                        optimizer=optimizer, checkpoint_dir=self.save_dir ,
                                                                            cleanup=True,add_stuffs=f"_member_{idx+1}")
                            best_losses[idx]  =  member_loss[idx]
                            
                            self.save_forecasts_to_csv(out_seq=results.expectations.squeeze(0) ,
                                                        var_names=[f"var_{i+1}" for i in range(model[f'member_{idx+1}'].n_obs_vars)],
                                                        forecast=forecasts_dict[f'member_{idx+1}'],filename=f"forecasts_epoch_{epoch+1}_member_{idx+1}_.csv")


                        print(f"✅ All ensemble members saved based on median loss: {float(np.median(member_loss)):.6f}")

                    else:
                        print(f"ℹ️ All ensemble members not saved based as median loss: {float(np.median(member_loss)):.6f} not better than best {float(np.median(best_losses)):.6f}. Skipping save.")

                    # epoch_loss_summary.append({f'epoch_{epoch+1}' : {'avg_train_loss': float(np.median(member_loss)),
                    #                                                   'median_train_loss': float(np.median(batch_losses)),
                    #                                                   'avg_val_loss': float(np.mean(batch_val_loss_summary)),
                    #                                                   'median_val_loss': float(np.median(batch_val_loss_summary))}})
                    epoch_loss_summary.append({f'epoch_{epoch+1}' : member_loss_summary})

                    print(f"[End of Epoch {epoch+1}  =================================================================")

                    self.ensembles =  self.load_ensemble_checkpoints(self.save_dir, train_type)

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