import torch
import os , json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Literal,Dict
from datetime import datetime
from torch.utils.data import DataLoader


from tqdm import tqdm



class EnsembleRNNTrainer:
    """
    Trainer for RNNEnsemble with per-model or ensemble-based saving.

    Parameters
    ----------
    ensemble : RNNEnsemble
        An ensemble of RNN_Model instances.

    optimizers : List[torch.optim.Optimizer]
        One optimizer per ensemble member.

    loss_fn : torch.nn.Module
        Loss function to use for training.

    best_on : Literal['individual', 'mean']
        Strategy to determine when to save models:
        - 'individual': Save each model based on its own best loss.
        - 'mean': Save all models based on the ensemble-wide median loss.
    """

    def __init__(self,
                 ensembles:Dict,
                #  optimizers: List[torch.optim.Optimizer],
                # optimizer:str,
                 loss_fn: torch.nn.Module,
                 best_on: Literal['individual', 'median'] = 'individual',
                 variable_names: Optional[List[str]] = None):

        self.ensembles = ensembles
        # self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_on = best_on
        self.variable_names = variable_names or [f"var_{i+1}" for i in range(ensembles[0][f"member_1"].output_size)]
        # self.device = ensemble.models[0]._get_default_device()

        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.save_dir = f"./checkpoints_{ensembles[0]['member_1'].name.split("_")[0]}ensemble_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)

    

    def train_only(self, data_loader: DataLoader, num_epochs: int = 10,
                   backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch') -> List[List[float]]:
        """
        Trains each ensemble member and saves based on training loss.

        Returns
        -------
        List of per-epoch training losses for each model.
        """
        all_losses = [[] for _ in self.ensembles]
        best_losses = [float('inf')] * len(self.ensembles)
        best_epoch = -1

        train_type = 'train_only'
        
        if self.best_on == 'individual':

            


            if backprop_mode == 'per_batch':
                epoch_losses = []
                for epoch in tqdm(range(num_epochs), desc="Training Only"):  
                    # epoch_losses = []
                    epoch_loss_summary ={}

                    for idx, model in enumerate(self.ensembles):
                        member_loss_summary = {}
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                        batch_idx = 0
                        member_batch_losses ={}
                        for batch_inps, batch_outs in data_loader:
                            
                            batch_size = batch_inps.size(0)
                            # h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()


                            h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()
                            # c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence=inps, initial_states=h0,
                                                                    batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            total_loss += loss.item()
                            batch_count += 1

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # member_loss_summary[f'member_{idx+1}'] = loss.item()
                            member_loss_summary[f'batch_idx_{batch_count+1}'] = {'train_loss':loss.item()}

                            if loss < best_losses[idx]:
                                    model[f'member_{idx+1}'].save_checkpoint(epoch+1, loss, optimizer,
                                                            checkpoint_dir=self.save_dir,
                                                            cleanup=True,
                                                            add_stuffs=f"_member_{idx+1}")
                                    best_losses[idx] = loss
                                    print(f"✅ [Model {idx+1}] New best train loss: {loss:.6f}")
                            else:
                                print(f"ℹ️ [Model {idx+1}] Train loss {loss:.6f} not better than best {best_losses[idx]:.6f}. Skipping save.")

                        batch_idx += 1
                        
                        epoch_loss_summary[f'member_{idx+1}'] = member_loss_summary

                    # avg_batch_loss = total_loss / batch_count

                    print(f"[End of Epoch {epoch+1}  =================================================================")
                          #)#| Model {idx+1}] Train Loss: {avg_batch_loss:.6f}")


                    # print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {avg_batch_loss:.6f}")
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir, train_type = train_type)
                                        
                
                    epoch_losses.append({ f'epoch {epoch+1}' : epoch_loss_summary})

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)
                # print(f"End of Epoch {epoch+1
                            # else:
                            #     accumulated_loss += loss

            elif backprop_mode == 'per_epoch':

                # epoch_losses = []
                    
                epoch_losses = []
                for epoch in tqdm(range(num_epochs), desc="Training Only"):  
                    
                    # epoch_losses = []
                    batch_idx = 0
                    # member_batch_losses ={}
                    # epoch_loss_summary ={}
                    member_loss_summary = {}
                    for idx, model in enumerate(self.ensembles):
                        
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()
                            # c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence =inps, 
                                                                   initial_states=h0, batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            # total_loss += loss.item()
                            batch_count += 1

                            accumulated_loss += loss

                        
                        
                    
                        total_loss= accumulated_loss/ batch_count
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        # member_batch_losses[f'member_{idx+1}'] = total_loss.item()

                        
                        # avg_loss = total_loss.item() #/ batch_count

                        member_loss_summary[f'member_{idx+1}'] = total_loss.item()
                        # epoch_losses.append({ 'epoch': f'{epoch+1}' , f'member_{idx+1}': total_loss.item()})
                        print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {total_loss:.6f}")

                        if total_loss < best_losses[idx]:
                            model[f'member_{idx+1}'].save_checkpoint(epoch+1, total_loss, optimizer,
                                                                    checkpoint_dir=self.save_dir,
                                                                    cleanup=True,
                                                                    add_stuffs=f"_member_{idx+1}")
                            best_losses[idx] = total_loss
                            print(f"✅ [Model {idx+1}] New best train loss: {total_loss:.6f}")
                        else:
                            print(f"ℹ️ [Model {idx+1}] Train loss {total_loss:.6f} not better than best {best_losses[idx]:.6f}. Skipping save.")
                    
                    # epoch_losses.append({ 'epoch': {epoch+1} , f'ensemble_losses' : member_batch_losses})
                        
                        # print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {avg_batch_loss:.6f}")
                    epoch_losses.append({ f'epoch {epoch+1}' : member_loss_summary}) 
                    print(f"[End of Epoch {epoch+1}  =================================================================")
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir , train_type = train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)


        elif self.best_on == 'median':


            if backprop_mode == 'per_batch':
                epoch_losses = []

                for epoch in tqdm(range(num_epochs), desc="Training Only"):  
                    epoch_loss_summary ={}
                    best_losses = [float('inf')] * len(self.ensembles)

                    # epoch_losses = []
                    batch_losses = []
                    batch_idx = 0


                    for batch_inps, batch_outs in data_loader:
                        batch_summary = []
                        # member_batch_losses = {}
                        member_loss_summary = {}
                        batch_count = 0
                        for idx, model in enumerate(self.ensembles):
                            
                            optimizer = model[f'optimizer_{idx+1}']
                            model[f'member_{idx+1}'].train()
                            total_loss, accumulated_loss = 0.0, 0.0
                            



                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()
                            # c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs


                            out = model[f'member_{idx+1}'].forward(input_sequence =inps, 
                                                                   initial_states=h0, 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            total_loss += loss.item()
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # member_batch_losses[f'member_{idx+1}'] = total_loss
                            # member_loss_summary[f'member_{idx+1}'] = {f'batch_idx_' {batch_count} , }   total_loss.item()
                            

                            batch_losses.append(total_loss)

                            member_loss_summary[f'member_{idx+1}'] = loss.item()
                        
                        if float(np.median(batch_losses)) < float(np.median(best_losses)):
                            for idx_, model in enumerate(self.ensembles):
                                model[f'member_{idx_+1}'].save_checkpoint(epoch+1, batch_losses[idx_],
                                                                        model[f'optimizer_{idx_+1}'],
                                                                        checkpoint_dir=self.save_dir,
                                                                        cleanup=True,
                                                                        add_stuffs=f"_member_{idx_+1}")
                                best_losses[idx_] = batch_losses[idx_]
                            print(f"✅ All Ensemble members saved based on median loss: {batch_losses[idx_]:.6f}")
                        else:
                            print(f"ℹ️ Median Ensemble train loss {float(np.median(batch_losses)):.6f} not better than best {float(np.median(best_losses)) :.6f}. Skipping save.")
                        #f'ensemble_median_{idx+1}': sum(batch_losses)/batch_count})
                        batch_count += 1
                        batch_summary.append({f'batch_idx_{batch_count}' :member_loss_summary})
                    epoch_losses.append({ f'epoch {epoch+1}' : member_loss_summary }) 

                    print(f"[End of Epoch {epoch+1} | Best Median loss across all models: {float(np.median(best_losses)):.6f}")

                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir, train_type = train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)



            elif backprop_mode == 'per_epoch':

                
                epoch_losses = []
                for epoch in tqdm(range(num_epochs), desc="Training Only"):  
                    best_losses = [float('inf')] * len(self.ensembles)
                    # epoch_losses = []
                    batch_losses = []

                    for idx, model in enumerate(self.ensembles):
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()
                            # c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence =inps,
                                                                   initial_states=h0, 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            # total_loss += loss.item()
                            batch_count += 1

                            accumulated_loss += loss
                    
                        total_loss= accumulated_loss/ batch_count
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        epoch_losses.append({ f'epoch {epoch+1}' : {f'member_{idx+1}': total_loss.item()}})

                        batch_losses.append(total_loss.item())

                    if float(np.median(batch_losses)) < float(np.median(best_losses)):
                            for idx_, model in enumerate(self.ensembles):
                                model[f'member_{idx_+1}'].save_checkpoint(epoch+1, batch_losses[idx_],
                                                                        model[f'optimizer_{idx_+1}'],
                                                                        checkpoint_dir=self.save_dir,
                                                                        cleanup=True,
                                                                        add_stuffs=f"_member_{idx_+1}")
                                best_losses = batch_losses[idx_]
                            print(f"✅ All Ensemble members saved based on median loss: {batch_losses[idx_]:.6f}")
                    else:
                            print(f"ℹ️ Median Ensemble train loss {float(np.median(batch_losses)):.6f} not better than best {float(np.median(best_losses)) :.6f}. Skipping save.")

                    print(f"[End of Epoch {epoch+1} | Median across all models] Train Loss: {float(np.median(best_losses)):.6f}")
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir , train_type = train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)


                
    def train_validate(self, data_loader: DataLoader, num_epochs: int ,
                       calibration_window: torch.Tensor,
                       val_data: torch.Tensor,
                   backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch') -> List[List[float]]:
        """
        Trains each ensemble member and saves based on training loss.

        Returns
        -------
        List of per-epoch training losses for each model.
        """
        all_losses = [[] for _ in self.ensembles]
        best_val_losses = [float('inf')] * len(self.ensembles)
        best_epoch = -1
        
        train_type = 'train_and_validate'
        if self.best_on == 'individual':


            if backprop_mode == 'per_batch':

                epoch_losses = []
                for epoch in tqdm(range(num_epochs), desc="Training and validating"):  
                    # epoch_losses = []
                    epoch_loss_summary = {} 
                    forecasts_dict = {}
                    for idx, model in enumerate(self.ensembles):
                        
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count ,batch_idx = 0.0, 0.0, 0 , 0
                        member_loss_summary ={}
                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()
                            # c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence = inps, 
                                                                   initial_states=h0, 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            total_loss += loss.item()
                            batch_count += 1

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()



                            model[f'member_{idx+1}'].eval() 
                            if model[f'member_{idx+1}'].ext_vars is not None:


                                
                                with torch.no_grad():

                                    out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                        calibration_window=calibration_window[:,:-model[f'member_{idx+1}'].ext_vars] , 
                                        n_steps=val_data.shape[0],
                                        externals_for_calibration=calibration_window[:,-model[f'member_{idx+1}'].ext_vars:] ,
                                        externals_for_forecasts=val_data[:,-model[f'member_{idx+1}'].ext_vars:])
                                    
                                    current_val_loss = self.loss_fn(forecasts, val_data[:,:-model[f'member_{idx+1}'].ext_vars]).item()
                                    
                            else:

                                with torch.no_grad():

                                    out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                        calibration_window=calibration_window, n_steps=val_data.shape[0])
                                    
                                    
                                    current_val_loss = self.loss_fn(forecasts, val_data).item()
        
                            forecasts_dict[f'member_{idx+1}'] = forecasts

                            if current_val_loss < best_val_losses[idx]:
                                
                                model[f'member_{idx+1}'].save_checkpoint(epoch+1, loss, optimizer,
                                                        checkpoint_dir=self.save_dir,
                                                        cleanup=True,
                                                        add_stuffs=f"_member_{idx+1}")
                                best_val_losses[idx] = current_val_loss
                                print(f"✅ [Model {idx+1}] New best val loss: {current_val_loss:.6f}")
                                self.save_forecasts_to_csv(out_seq=out_seq, var_names=[f"var_{i+1}" for i in range(model[f'member_{idx+1}'].output_size)], 
                                                            forecast=forecasts_dict[f'member_{idx+1}'] , filename=f"forecasts_epoch_{epoch+1}_member_{idx+1}_.csv")
                            else:
                                print(f"ℹ️ [Model {idx+1}] cureent val_loss {current_val_loss:.6f} not better than best val_loss: {best_val_losses[idx]:.6f}. Skipping save.")

                            avg_batch_val_loss = total_loss / batch_count


                            member_loss_summary[f'batch_idx_{batch_count+1}'] = {'train_loss':loss.item(), 'val_loss': current_val_loss}
                        
                            batch_idx+=1

                        epoch_loss_summary[f'member_{idx+1}'] = {f'epoch_{epoch+1}': member_loss_summary}
                        print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {avg_batch_val_loss:.6f}")

                    epoch_losses.append({ f'epoch{epoch+1}' : epoch_loss_summary})
                    
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir , train_type = train_type)
                
                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)

                    
                            # else:
                            #     accumulated_loss += loss

            elif backprop_mode == 'per_epoch':
                    
                epoch_losses = []
                for epoch in tqdm(range(num_epochs), desc="Training and validating"):  
                    forecasts_dict={}
                    batch_losses = []
                    epoch_loss_summary = {}
                    for idx, model in enumerate(self.ensembles):
                        member_loss_summary = {}
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()
                            # c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence = inps, 
                                                                   initial_states=h0, 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            # total_loss += loss.item()
                            batch_count += 1

                            accumulated_loss += loss
                    
                        total_loss= accumulated_loss/ batch_count
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        # avg_loss = total_loss.item() #/ batch_count
                        # epoch_losses.append(total_loss.item())


                        model[f'member_{idx+1}'].eval() 
                        if model[f'member_{idx+1}'].ext_vars is not None:


                            
                            with torch.no_grad():

                                out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                    calibration_window=calibration_window[:,:-model[f'member_{idx+1}'].ext_vars] , 
                                    n_steps=val_data.shape[0],
                                    externals_for_calibration=calibration_window[:,-model[f'member_{idx+1}'].ext_vars:] ,
                                    externals_for_forecasts=val_data[:,-model[f'member_{idx+1}'].ext_vars:])
                                
                                current_val_loss = self.loss_fn(forecasts, val_data[:,:-model[f'member_{idx+1}'].ext_vars]).item()
                                
                        else:

                            with torch.no_grad():

                                out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                    calibration_window=calibration_window, n_steps=val_data.shape[0])
                                
                        forecasts_dict[f'member_{idx+1}'] = forecasts
                        current_val_loss = self.loss_fn(forecasts, val_data).item()
                        member_loss_summary[f'member_{idx+1}'] = {'train_loss':total_loss.item(), 'val_loss': current_val_loss}

                        if current_val_loss < best_val_losses[idx]:
                                    model[f'member_{idx+1}'].save_checkpoint(epoch+1, loss, optimizer,
                                                            checkpoint_dir=self.save_dir,
                                                            cleanup=True,
                                                            add_stuffs=f"_member_{idx+1}")
                                    best_val_losses[idx] = current_val_loss
                                    print(f"✅ [Model {idx+1}] New best val loss: {current_val_loss:.6f}")
                                    self.save_forecasts_to_csv(out_seq=out_seq, var_names=[f"var_{i+1}" for i in range(model[f'member_{idx+1}'].output_size)], 
                                                               forecast=forecasts_dict[f'member_{idx+1}'] , filename=f"forecasts_epoch_{epoch+1}_member_{idx+1}_.csv")
                        else:
                            print(f"ℹ️ [Model {idx+1}] current val_loss {current_val_loss:.6f} not better than best val_loss: {best_val_losses[idx]:.6f}. Skipping save.")

                        
                    epoch_loss_summary[f'member_{idx+1}'] = member_loss_summary

 
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir, train_type = train_type)
                
                epoch_losses.append({ f'epoch{epoch+1}' : epoch_loss_summary})
                print(f"[End of Epoch {epoch+1} | Best Median loss across all models: {float(np.median(best_val_losses)):.6f}")


                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)


        elif self.best_on == 'median':


            if backprop_mode == 'per_batch':

                epoch_losses = []

                for epoch in tqdm(range(num_epochs), desc="Training and validating"):  
                    forecasts_dict={}
                    best_losses = [float('inf')] * len(self.ensembles)
                    epoch_loss_summary = {}

                    # epoch_losses = []
                    batch_losses = []
                    for batch_inps, batch_outs in data_loader:
                        member_loss_summary = {}
                        member_val_losses = []
                        for idx, model in enumerate(self.ensembles):
                            optimizer = model[f'optimizer_{idx+1}']
                            model[f'member_{idx+1}'].train()
                            total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                            



                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()
                            # c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs


                            out = model[f'member_{idx+1}'].forward(input_sequence =inps, 
                                                                   initial_states=h0, 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            total_loss += loss.item()
                            batch_count += 1
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            batch_losses.append(loss.item())

                            model[f'member_{idx+1}'].eval()
                            if model[f'member_{idx+1}'].ext_vars is not None:


                                    
                                    with torch.no_grad():

                                        out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                            calibration_window=calibration_window[:,:-model[f'member_{idx+1}'].ext_vars] , 
                                            n_steps=val_data.shape[0],
                                            externals_for_calibration=calibration_window[:,-model[f'member_{idx+1}'].ext_vars:] ,
                                            externals_for_forecasts=val_data[:,-model[f'member_{idx+1}'].ext_vars:])
                                        
                                        current_val_loss = self.loss_fn(forecasts, val_data[:,:-model[f'member_{idx+1}'].ext_vars]).item()
                                        
                            else:

                                with torch.no_grad():

                                    out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                        calibration_window=calibration_window, n_steps=val_data.shape[0])
                                    
                                    
                                    current_val_loss = self.loss_fn(forecasts, val_data).item()
                            forecasts_dict[f'member_{idx+1}'] = forecasts
                            member_val_losses.append(current_val_loss)

                            member_loss_summary[f'member_{idx+1}'] = {'train_loss':loss.item(), 'val_loss': current_val_loss}

                        if float(np.median(member_val_losses)) < float(np.median(best_losses)):

                            for idx_, model in enumerate(self.ensembles):
                                
                                model[f'member_{idx_+1}'].save_checkpoint(epoch+1, member_val_losses[idx_],
                                                                        model[f'optimizer_{idx_+1}'],
                                                                        checkpoint_dir=self.save_dir,
                                                                        cleanup=True,
                                                                        add_stuffs=f"_member_{idx_+1}")
                                
                                self.save_forecasts_to_csv(out_seq=out_seq, var_names=[f"var_{i+1}" for i in range(model[f'member_{idx_+1}'].output_size)], 
                                                                forecast=forecasts_dict[f'member_{idx_+1}'], filename=f"forecasts_epoch_{epoch+1}_member_{idx_+1}_.csv")
                                
                                best_losses[idx_] = member_val_losses[idx_]

                                
                            print(f"✅ All Ensemble members saved based on median loss: {float(np.median(member_val_losses)):.6f}")

                        else:
                            print(f"ℹ️ Median Ensemble train loss {float(np.median(member_val_losses)):.6f} not better than best {float(np.median(best_losses)) :.6f}. Skipping save.")





                        epoch_losses.append({ f'epoch{epoch+1}' : member_loss_summary})
                    # if total_loss < best_losses[idx]:
                    print(f"[End of Epoch {epoch+1} | Best Median loss across all models: {float(np.median(best_losses)):.6f}")

                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir, train_type = train_type)
                
                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)



            elif backprop_mode == 'per_epoch':

                epoch_losses = []

                # forecasts_dict={}
                    
                for epoch in tqdm(range(num_epochs), desc="Training and validating"):  
                    forecasts_dict={}
                    best_losses = [float('inf')] * len(self.ensembles)
                    # epoch_losses = []
                    batch_losses = []
                    epoch_loss_summary = {}
                    member_loss_summary = {}
                    member_val_losses = []

                    for idx, model in enumerate(self.ensembles):
                        
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()
                            # c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence = inps, 
                                                                   initial_states  =h0,
                                                                     batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            # total_loss += loss.item()
                            batch_count += 1

                            accumulated_loss += loss
                    
                        total_loss= accumulated_loss/ batch_count
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()


                        batch_losses.append(total_loss.item())



                        model[f'member_{idx+1}'].eval()
                        if model[f'member_{idx+1}'].ext_vars is not None:


                                
                                with torch.no_grad():

                                    out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                        calibration_window=calibration_window[:,:-model[f'member_{idx+1}'].ext_vars] , 
                                        n_steps=val_data.shape[0],
                                        externals_for_calibration=calibration_window[:,-model[f'member_{idx+1}'].ext_vars:] ,
                                        externals_for_forecasts=val_data[:,-model[f'member_{idx+1}'].ext_vars:])
                                    
                                    current_val_loss = self.loss_fn(forecasts, val_data[:,:-model[f'member_{idx+1}'].ext_vars]).item()
                                    
                        else:

                            with torch.no_grad():

                                out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                    calibration_window=calibration_window, n_steps=val_data.shape[0])
                                
                                
                                current_val_loss = self.loss_fn(forecasts, val_data).item()

                        forecasts_dict[f'member_{idx+1}'] = forecasts
                        member_val_losses.append(current_val_loss)
                        member_loss_summary[f'member_{idx+1}'] = {'train_loss':total_loss.item(), 'val_loss': current_val_loss}
                    
                    if float(np.median(member_val_losses)) < float(np.median(best_losses)):
                        
                        for idx_, model in enumerate(self.ensembles):

                            model[f'member_{idx_+1}'].save_checkpoint(epoch+1, member_val_losses[idx_],
                                                                    model[f'optimizer_{idx_+1}'],
                                                                    checkpoint_dir=self.save_dir,
                                                                    cleanup=True,
                                                                    add_stuffs=f"_member_{idx_+1}")
                            
                            best_losses[idx_] = member_val_losses[idx_]

                            self.save_forecasts_to_csv(out_seq=out_seq, var_names=[f"var_{i+1}" for i in range(model[f'member_{idx_+1}'].output_size)], 
                                                                forecast=forecasts_dict[f'member_{idx_+1}'], filename=f"forecasts_epoch_{epoch+1}_member_{idx_+1}_.csv")
                            print (f"✅ [Model {idx_+1}] New best val loss: {member_val_losses[idx_]:.6f} | forecast results saved")
                        print(f"✅ All Ensemble members saved based on median loss: {np.median(member_val_losses):.6f}")
                    else:
                        print(f"ℹ️ Median Ensemble train loss {float(np.median(member_val_losses)):.6f} not better than best {float(np.median(best_losses)) :.6f}. Skipping save.")

                # epoch_loss_summary
                    epoch_losses.append({ f'epoch{epoch+1}' :member_loss_summary})

                    print(f"[End of Epoch {epoch+1} | Best Median loss across all models: {float(np.median(best_losses)):.6f}")


                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir , train_type = train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)



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

    def train_forecast(self, data_loader: DataLoader,
                       calibration_window: torch.Tensor,
                       val_data: torch.Tensor,
                       num_epochs: int = 10,
                       backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch') -> List[List[float]]:
        """
        Trains ensemble and saves based on validation loss.

        Returns
        -------
        List of per-epoch training losses for each model.
        """
        all_losses = [[] for _ in range(len(self.ensembles))]
        best_losses = [float('inf')] * len(self.ensembles)
        best_epoch = -1

        for epoch in range(num_epochs):
            epoch_losses = []
            forecasts_for_epoch = []

            for idx, model in enumerate(self.ensembles):
                optimizer = model[f'optimizer_{idx+1}']
                model[f'member_{idx+1}'].train()

                total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                for batch_inps, batch_outs in data_loader:
                    batch_size = batch_inps.size(0)
                    h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()

                    if model[f'member_{idx+1}'].ext_vars:
                        inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                        exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                        targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                    else:
                        inps, exts, targets = batch_inps, None, batch_outs

                    out = model[f'member_{idx+1}'].forward(input_sequence =inps, 
                                                           initial_states=h0, batch_of_externals=exts)
                    loss = self.loss_fn(out.output_sequence, targets)
                    total_loss += loss.item()
                    batch_count += 1

                    if backprop_mode == 'per_batch':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        accumulated_loss += loss

                if backprop_mode == 'per_epoch':
                    optimizer.zero_grad()
                    accumulated_loss.backward()
                    optimizer.step()

                avg_loss = total_loss / batch_count
                all_losses[idx].append(avg_loss)



                # Validation forecast
                model[f'member_{idx+1}'].eval()
                with torch.no_grad():
                    if model[f'member_{idx+1}'].ext_vars:
                        out_seq, forecast = model[f'member_{idx+1}'].forecast_seq_to_seq(
                            calibration_window[:, :-model[f'member_{idx+1}'].ext_vars],
                            n_steps=val_data.shape[0],
                            externals_for_calibration=calibration_window[:, -model[f'member_{idx+1}'].ext_vars:],
                            externals_for_forecasts=val_data[:, -model[f'member_{idx+1}'].ext_vars:]
                        )
                        val_loss = self.loss_fn(forecast, val_data[:, :-model[f'member_{idx+1}'].ext_vars]).item()
                    else:
                        out_seq, forecast = model[f'member_{idx+1}'].forecast_seq_to_seq(calibration_window, n_steps=val_data.shape[0])
                        val_loss = self.loss_fn(forecast, val_data).item()

                    epoch_losses.append(val_loss)
                    forecasts_for_epoch.append((out_seq, forecast))

                print(f"[Epoch {epoch+1} | Model {idx+1}] Val Loss: {val_loss:.6f}")

                # Saving logic
                if self.best_on == 'individual':
                    for idx, loss in enumerate(epoch_losses):
                        if loss < best_losses[idx]:
                            model[f'member_{idx+1}'].save_checkpoint(epoch+1, loss, model[f'optimizer_{idx+1}'],
                                                                    checkpoint_dir=self.save_dir, cleanup=True)
                            self._save_forecasts_to_csv(*forecasts_for_epoch[idx], idx, epoch)
                            best_losses[idx] = loss
                            print(f"✅ [Model {idx+1}] New best val loss: {loss:.6f}")
        if self.best_on == 'median':
            median_loss = float(torch.tensor(epoch_losses).median())
            # median_loss = torch.median(torch.tensor(epoch_losses)).item()
            if median_loss < min(best_losses):
                for idx in range(self.ensemble.n_ensemble):
                    model[f'member_{idx+1}'].save_checkpoint(epoch+1, epoch_losses[idx], model[f'optimizer_{idx+1}'],
                                                                checkpoint_dir=self.save_dir, cleanup=True)
                    self._save_forecasts_to_csv(*forecasts_for_epoch[idx], idx, epoch)
                    best_losses[idx] = epoch_losses[idx]
                best_epoch = epoch + 1
                print(f"✅ Ensemble saved based on median val loss: {median_loss:.6f}")



        return all_losses


    def _save_losses_json(self, model, losses: List[float], model_idx: int):
        path = os.path.join(self.save_dir, f"losses_model{model_idx+1}_{model.name}.json")
        with open(path, 'w') as f:
            json.dump([{"epoch": i+1, "avg_loss": l} for i, l in enumerate(losses)], f, indent=2)


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
                checkpoint = torch.load(checkpoint_path_, map_location=self.ensembles[idx][f'member_{idx+1}']._get_default_device())
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
                checkpoint = torch.load(checkpoint_path_, map_location=self.ensembles[idx][f'member_{idx+1}']._get_default_device())
                self.ensembles[idx][f'member_{idx+1}'].load_state_dict(checkpoint["model_state_dict"])
                self.ensembles[idx][f'optimizer_{idx+1}'].load_state_dict(checkpoint["optimizer_state_dict"])

                print(f"✅ Loaded Model {idx+1} from {checkpoint_path_} | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.6f}")

            



        return self.ensembles
    

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





class EnsembleLSTMTrainer:
    """
    Trainer for RNNEnsemble with per-model or ensemble-based saving.

    Parameters
    ----------
    ensemble : RNNEnsemble
        An ensemble of RNN_Model instances.

    optimizers : List[torch.optim.Optimizer]
        One optimizer per ensemble member.

    loss_fn : torch.nn.Module
        Loss function to use for training.

    best_on : Literal['individual', 'mean']
        Strategy to determine when to save models:
        - 'individual': Save each model based on its own best loss.
        - 'mean': Save all models based on the ensemble-wide median loss.
    """

    def __init__(self,
                 ensembles:Dict,
                #  optimizers: List[torch.optim.Optimizer],
                # optimizer:str,
                 loss_fn: torch.nn.Module,
                 best_on: Literal['individual', 'median'] = 'individual',
                 variable_names: Optional[List[str]] = None):

        self.ensembles = ensembles
        # self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_on = best_on
        self.variable_names = variable_names or [f"var_{i+1}" for i in range(ensembles[0][f"member_1"].output_size)]
        # self.device = ensemble.models[0]._get_default_device()

        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.save_dir = f"./checkpoints_{ensembles[0]['member_1'].name.split("_")[0]}ensemble_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)

    

    def train_only(self, data_loader: DataLoader, num_epochs: int = 10,
                   backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch') -> List[List[float]]:
        """
        Trains each ensemble member and saves based on training loss.

        Returns
        -------
        List of per-epoch training losses for each model.
        """
        all_losses = [[] for _ in self.ensembles]
        best_losses = [float('inf')] * len(self.ensembles)
        best_epoch = -1

        train_type = 'train_only'
        
        if self.best_on == 'individual':

            


            if backprop_mode == 'per_batch':
                epoch_losses = []
                for epoch in range(num_epochs):
                    # epoch_losses = []
                    epoch_loss_summary ={}

                    for idx, model in enumerate(self.ensembles):
                        member_loss_summary = {}
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                        batch_idx = 0
                        member_batch_losses ={}
                        for batch_inps, batch_outs in data_loader:
                            
                            batch_size = batch_inps.size(0)
                            # h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()


                            h0 = model[f'member_{idx+1}'].initial_h.expand(-1, batch_size, -1).contiguous()
                            c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence=inps, initial_states=(h0,c0),
                                                                    batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            total_loss += loss.item()
                            batch_count += 1

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # member_loss_summary[f'member_{idx+1}'] = loss.item()
                            member_loss_summary[f'batch_idx_{batch_count+1}'] = {'train_loss':loss.item()}

                            if loss < best_losses[idx]:
                                    model[f'member_{idx+1}'].save_checkpoint(epoch+1, loss, optimizer,
                                                            checkpoint_dir=self.save_dir,
                                                            cleanup=True,
                                                            add_stuffs=f"_member_{idx+1}")
                                    best_losses[idx] = loss
                                    print(f"✅ [Model {idx+1}] New best train loss: {loss:.6f}")
                            else:
                                print(f"ℹ️ [Model {idx+1}] Train loss {loss:.6f} not better than best {best_losses[idx]:.6f}. Skipping save.")

                        batch_idx += 1
                        
                        epoch_loss_summary[f'member_{idx+1}'] = member_loss_summary

                    # avg_batch_loss = total_loss / batch_count

                    print(f"[End of Epoch {epoch+1}  =================================================================")
                          #)#| Model {idx+1}] Train Loss: {avg_batch_loss:.6f}")


                    # print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {avg_batch_loss:.6f}")
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir, train_type = train_type)
                                        
                
                    epoch_losses.append({ f'epoch {epoch+1}' : epoch_loss_summary})

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)
                # print(f"End of Epoch {epoch+1
                            # else:
                            #     accumulated_loss += loss

            elif backprop_mode == 'per_epoch':

                # epoch_losses = []
                    
                epoch_losses = []
                for epoch in range(num_epochs):
                    
                    # epoch_losses = []
                    batch_idx = 0
                    # member_batch_losses ={}
                    # epoch_loss_summary ={}
                    member_loss_summary = {}
                    for idx, model in enumerate(self.ensembles):
                        
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_h.expand(-1, batch_size, -1).contiguous()
                            c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence =inps, 
                                                                   initial_states=(h0,c0), batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            # total_loss += loss.item()
                            batch_count += 1

                            accumulated_loss += loss

                        
                        
                    
                        total_loss= accumulated_loss/ batch_count
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()


                        member_loss_summary[f'member_{idx+1}'] = total_loss.item()
                        # epoch_losses.append({ 'epoch': f'{epoch+1}' , f'member_{idx+1}': total_loss.item()})
                        print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {total_loss:.6f}")

                        if total_loss < best_losses[idx]:
                            model[f'member_{idx+1}'].save_checkpoint(epoch+1, total_loss, optimizer,
                                                                    checkpoint_dir=self.save_dir,
                                                                    cleanup=True,
                                                                    add_stuffs=f"_member_{idx+1}")
                            best_losses[idx] = total_loss
                            print(f"✅ [Model {idx+1}] New best train loss: {total_loss:.6f}")
                        else:
                            print(f"ℹ️ [Model {idx+1}] Train loss {total_loss:.6f} not better than best {best_losses[idx]:.6f}. Skipping save.")
                    

                        
                        # print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {avg_batch_loss:.6f}")
                    epoch_losses.append({ f'epoch {epoch+1}' : member_loss_summary}) 
                    print(f"[End of Epoch {epoch+1}  =================================================================")
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir , train_type = train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)


        elif self.best_on == 'median':


            if backprop_mode == 'per_batch':
                epoch_losses = []

                for epoch in range(num_epochs):
                    epoch_loss_summary ={}
                    best_losses = [float('inf')] * len(self.ensembles)

                    # epoch_losses = []
                    batch_losses = []
                    batch_idx = 0


                    for batch_inps, batch_outs in data_loader:
                        batch_summary = []
                        # member_batch_losses = {}
                        member_loss_summary = {}
                        batch_count = 0
                        for idx, model in enumerate(self.ensembles):
                            
                            optimizer = model[f'optimizer_{idx+1}']
                            model[f'member_{idx+1}'].train()
                            total_loss, accumulated_loss = 0.0, 0.0
                            



                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_h.expand(-1, batch_size, -1).contiguous()
                            c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs


                            out = model[f'member_{idx+1}'].forward(input_sequence =inps, 
                                                                   initial_states=(h0,c0), 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            total_loss += loss.item()
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            # member_batch_losses[f'member_{idx+1}'] = total_loss
                            # member_loss_summary[f'member_{idx+1}'] = {f'batch_idx_' {batch_count} , }   total_loss.item()
                            

                            batch_losses.append(total_loss)

                            member_loss_summary[f'member_{idx+1}'] = loss.item()
                        
                        if float(np.median(batch_losses)) < float(np.median(best_losses)):
                            for idx_, model in enumerate(self.ensembles):
                                model[f'member_{idx_+1}'].save_checkpoint(epoch+1, batch_losses[idx_],
                                                                        model[f'optimizer_{idx_+1}'],
                                                                        checkpoint_dir=self.save_dir,
                                                                        cleanup=True,
                                                                        add_stuffs=f"_member_{idx_+1}")
                                best_losses[idx_] = batch_losses[idx_]
                            print(f"✅ All Ensemble members saved based on median loss: {batch_losses[idx_]:.6f}")
                        else:
                            print(f"ℹ️ Median Ensemble train loss {float(np.median(batch_losses)):.6f} not better than best {float(np.median(best_losses)) :.6f}. Skipping save.")
                        #f'ensemble_median_{idx+1}': sum(batch_losses)/batch_count})
                        batch_count += 1
                        batch_summary.append({f'batch_idx_{batch_count}' :member_loss_summary})
                    epoch_losses.append({ f'epoch {epoch+1}' : member_loss_summary }) 

                    print(f"[End of Epoch {epoch+1} | Best Median loss across all models: {float(np.median(best_losses)):.6f}")

                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir, train_type = train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)



            elif backprop_mode == 'per_epoch':

                
                epoch_losses = []
                for epoch in range(num_epochs):
                    best_losses = [float('inf')] * len(self.ensembles)
                    # epoch_losses = []
                    batch_losses = []

                    for idx, model in enumerate(self.ensembles):
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_h.expand(-1, batch_size, -1).contiguous()
                            c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence =inps,
                                                                   initial_states=(h0,c0), 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            # total_loss += loss.item()
                            batch_count += 1

                            accumulated_loss += loss
                    
                        total_loss= accumulated_loss/ batch_count
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        epoch_losses.append({ f'epoch {epoch+1}' : {f'member_{idx+1}': total_loss.item()}})

                        batch_losses.append(total_loss.item())

                    if float(np.median(batch_losses)) < float(np.median(best_losses)):
                            for idx_, model in enumerate(self.ensembles):
                                model[f'member_{idx_+1}'].save_checkpoint(epoch+1, batch_losses[idx_],
                                                                        model[f'optimizer_{idx_+1}'],
                                                                        checkpoint_dir=self.save_dir,
                                                                        cleanup=True,
                                                                        add_stuffs=f"_member_{idx_+1}")
                                best_losses = batch_losses[idx_]
                            print(f"✅ All Ensemble members saved based on median loss: {batch_losses[idx_]:.6f}")
                    else:
                            print(f"ℹ️ Median Ensemble train loss {float(np.median(batch_losses)):.6f} not better than best {float(np.median(best_losses)) :.6f}. Skipping save.")

                    print(f"[End of Epoch {epoch+1} | Median across all models] Train Loss: {float(np.median(best_losses)):.6f}")
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir , train_type = train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)


                
    def train_validate(self, data_loader: DataLoader, num_epochs: int ,
                       calibration_window: torch.Tensor,
                       val_data: torch.Tensor,
                   backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch') -> List[List[float]]:
        """
        Trains each ensemble member and saves based on training loss.

        Returns
        -------
        List of per-epoch training losses for each model.
        """
        all_losses = [[] for _ in self.ensembles]
        best_val_losses = [float('inf')] * len(self.ensembles)
        best_epoch = -1
        
        train_type = 'train_and_validate'
        if self.best_on == 'individual':


            if backprop_mode == 'per_batch':

                epoch_losses = []
                for epoch in range(num_epochs):
                    # epoch_losses = []
                    epoch_loss_summary = {} 
                    forecasts_dict = {}
                    for idx, model in enumerate(self.ensembles):
                        
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count ,batch_idx = 0.0, 0.0, 0 , 0
                        member_loss_summary ={}
                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_h.expand(-1, batch_size, -1).contiguous()
                            c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence = inps, 
                                                                   initial_states=(h0,c0), 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            total_loss += loss.item()
                            batch_count += 1

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()



                            model[f'member_{idx+1}'].eval() 
                            if model[f'member_{idx+1}'].ext_vars is not None:


                                
                                with torch.no_grad():

                                    out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                        calibration_window=calibration_window[:,:-model[f'member_{idx+1}'].ext_vars] , 
                                        n_steps=val_data.shape[0],
                                        externals_for_calibration=calibration_window[:,-model[f'member_{idx+1}'].ext_vars:] ,
                                        externals_for_forecasts=val_data[:,-model[f'member_{idx+1}'].ext_vars:])
                                    
                                    current_val_loss = self.loss_fn(forecasts, val_data[:,:-model[f'member_{idx+1}'].ext_vars]).item()
                                    
                            else:

                                with torch.no_grad():

                                    out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                        calibration_window=calibration_window, n_steps=val_data.shape[0])
                                    
                                    
                                    current_val_loss = self.loss_fn(forecasts, val_data).item()
        
                            forecasts_dict[f'member_{idx+1}'] = forecasts

                            if current_val_loss < best_val_losses[idx]:
                                
                                model[f'member_{idx+1}'].save_checkpoint(epoch+1, loss, optimizer,
                                                        checkpoint_dir=self.save_dir,
                                                        cleanup=True,
                                                        add_stuffs=f"_member_{idx+1}")
                                best_val_losses[idx] = current_val_loss
                                print(f"✅ [Model {idx+1}] New best val loss: {current_val_loss:.6f}")
                                self.save_forecasts_to_csv(out_seq=out_seq, var_names=[f"var_{i+1}" for i in range(model[f'member_{idx+1}'].output_size)], 
                                                            forecast=forecasts_dict[f'member_{idx+1}'] , filename=f"forecasts_epoch_{epoch+1}_member_{idx+1}_.csv")
                            else:
                                print(f"ℹ️ [Model {idx+1}] cureent val_loss {current_val_loss:.6f} not better than best val_loss: {best_val_losses[idx]:.6f}. Skipping save.")

                            avg_batch_val_loss = total_loss / batch_count


                            member_loss_summary[f'batch_idx_{batch_count+1}'] = {'train_loss':loss.item(), 'val_loss': current_val_loss}
                        
                            batch_idx+=1

                        epoch_loss_summary[f'member_{idx+1}'] = {f'epoch_{epoch+1}': member_loss_summary}
                        print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {avg_batch_val_loss:.6f}")

                    epoch_losses.append({ f'epoch{epoch+1}' : epoch_loss_summary})
                    
                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir , train_type = train_type)
                
                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)

                    
                            # else:
                            #     accumulated_loss += loss

            elif backprop_mode == 'per_epoch':
                    
                epoch_losses = []
                for epoch in range(num_epochs):
                    forecasts_dict={}
                    batch_losses = []
                    epoch_loss_summary = {}
                    for idx, model in enumerate(self.ensembles):
                        member_loss_summary = {}
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_h.expand(-1, batch_size, -1).contiguous()
                            c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence = inps, 
                                                                   initial_states=(h0,c0), 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            # total_loss += loss.item()
                            batch_count += 1

                            accumulated_loss += loss
                    
                        total_loss= accumulated_loss/ batch_count
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        # avg_loss = total_loss.item() #/ batch_count
                        # epoch_losses.append(total_loss.item())


                        model[f'member_{idx+1}'].eval() 
                        if model[f'member_{idx+1}'].ext_vars is not None:


                            
                            with torch.no_grad():

                                out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                    calibration_window=calibration_window[:,:-model[f'member_{idx+1}'].ext_vars] , 
                                    n_steps=val_data.shape[0],
                                    externals_for_calibration=calibration_window[:,-model[f'member_{idx+1}'].ext_vars:] ,
                                    externals_for_forecasts=val_data[:,-model[f'member_{idx+1}'].ext_vars:])
                                
                                current_val_loss = self.loss_fn(forecasts, val_data[:,:-model[f'member_{idx+1}'].ext_vars]).item()
                                
                        else:

                            with torch.no_grad():

                                out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                    calibration_window=calibration_window, n_steps=val_data.shape[0])
                                
                        forecasts_dict[f'member_{idx+1}'] = forecasts
                        current_val_loss = self.loss_fn(forecasts, val_data).item()
                        member_loss_summary[f'member_{idx+1}'] = {'train_loss':total_loss.item(), 'val_loss': current_val_loss}
                        # batch_losses.append(total_loss.item())
                        # print(f"[Epoch {epoch+1} | Model {idx+1}] Train Loss: {total_loss:.6f}")
                        if current_val_loss < best_val_losses[idx]:
                                    model[f'member_{idx+1}'].save_checkpoint(epoch+1, loss, optimizer,
                                                            checkpoint_dir=self.save_dir,
                                                            cleanup=True,
                                                            add_stuffs=f"_member_{idx+1}")
                                    best_val_losses[idx] = current_val_loss
                                    print(f"✅ [Model {idx+1}] New best val loss: {current_val_loss:.6f}")
                                    self.save_forecasts_to_csv(out_seq=out_seq, var_names=[f"var_{i+1}" for i in range(model[f'member_{idx+1}'].output_size)], 
                                                               forecast=forecasts_dict[f'member_{idx+1}'] , filename=f"forecasts_epoch_{epoch+1}_member_{idx+1}_.csv")
                        else:
                            print(f"ℹ️ [Model {idx+1}] current val_loss {current_val_loss:.6f} not better than best val_loss: {best_val_losses[idx]:.6f}. Skipping save.")

                        
                    epoch_loss_summary[f'member_{idx+1}'] = member_loss_summary


                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir, train_type = train_type)
                
                epoch_losses.append({ f'epoch{epoch+1}' : epoch_loss_summary})
                print(f"[End of Epoch {epoch+1} | Best Median loss across all models: {float(np.median(best_val_losses)):.6f}")


                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)


        elif self.best_on == 'median':


            if backprop_mode == 'per_batch':

                epoch_losses = []

                for epoch in range(num_epochs):
                    forecasts_dict={}
                    best_losses = [float('inf')] * len(self.ensembles)
                    epoch_loss_summary = {}

                    # epoch_losses = []
                    batch_losses = []
                    for batch_inps, batch_outs in data_loader:
                        member_loss_summary = {}
                        member_val_losses = []
                        for idx, model in enumerate(self.ensembles):
                            optimizer = model[f'optimizer_{idx+1}']
                            model[f'member_{idx+1}'].train()
                            total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0
                            



                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_h.expand(-1, batch_size, -1).contiguous()
                            c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs


                            out = model[f'member_{idx+1}'].forward(input_sequence =inps, 
                                                                   initial_states=(h0,c0), 
                                                                   batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            total_loss += loss.item()
                            batch_count += 1
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            batch_losses.append(loss.item())

                            model[f'member_{idx+1}'].eval()
                            if model[f'member_{idx+1}'].ext_vars is not None:


                                    
                                    with torch.no_grad():

                                        out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                            calibration_window=calibration_window[:,:-model[f'member_{idx+1}'].ext_vars] , 
                                            n_steps=val_data.shape[0],
                                            externals_for_calibration=calibration_window[:,-model[f'member_{idx+1}'].ext_vars:] ,
                                            externals_for_forecasts=val_data[:,-model[f'member_{idx+1}'].ext_vars:])
                                        
                                        current_val_loss = self.loss_fn(forecasts, val_data[:,:-model[f'member_{idx+1}'].ext_vars]).item()
                                        
                            else:

                                with torch.no_grad():

                                    out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                        calibration_window=calibration_window, n_steps=val_data.shape[0])
                                    
                                    
                                    current_val_loss = self.loss_fn(forecasts, val_data).item()
                            forecasts_dict[f'member_{idx+1}'] = forecasts
                            member_val_losses.append(current_val_loss)

                            member_loss_summary[f'member_{idx+1}'] = {'train_loss':loss.item(), 'val_loss': current_val_loss}

                        if float(np.median(member_val_losses)) < float(np.median(best_losses)):

                            for idx_, model in enumerate(self.ensembles):
                                
                                model[f'member_{idx_+1}'].save_checkpoint(epoch+1, member_val_losses[idx_],
                                                                        model[f'optimizer_{idx_+1}'],
                                                                        checkpoint_dir=self.save_dir,
                                                                        cleanup=True,
                                                                        add_stuffs=f"_member_{idx_+1}")
                                
                                self.save_forecasts_to_csv(out_seq=out_seq, var_names=[f"var_{i+1}" for i in range(model[f'member_{idx_+1}'].output_size)], 
                                                                forecast=forecasts_dict[f'member_{idx_+1}'], filename=f"forecasts_epoch_{epoch+1}_member_{idx_+1}_.csv")
                                
                                best_losses[idx_] = member_val_losses[idx_]

                                
                            print(f"✅ All Ensemble members saved based on median loss: {float(np.median(member_val_losses)):.6f}")

                        else:
                            print(f"ℹ️ Median Ensemble train loss {float(np.median(member_val_losses)):.6f} not better than best {float(np.median(best_losses)) :.6f}. Skipping save.")





                        epoch_losses.append({ f'epoch{epoch+1}' : member_loss_summary})
                    # if total_loss < best_losses[idx]:
                    print(f"[End of Epoch {epoch+1} | Best Median loss across all models: {float(np.median(best_losses)):.6f}")

                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir, train_type = train_type)
                
                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)



            elif backprop_mode == 'per_epoch':

                epoch_losses = []

                # forecasts_dict={}
                    
                for epoch in range(num_epochs):
                    forecasts_dict={}
                    best_losses = [float('inf')] * len(self.ensembles)
                    # epoch_losses = []
                    batch_losses = []
                    epoch_loss_summary = {}
                    member_loss_summary = {}
                    member_val_losses = []

                    for idx, model in enumerate(self.ensembles):
                        
                        optimizer = model[f'optimizer_{idx+1}']
                        model[f'member_{idx+1}'].train()
                        total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                        for batch_inps, batch_outs in data_loader:
                            batch_size = batch_inps.size(0)
                            h0 = model[f'member_{idx+1}'].initial_h.expand(-1, batch_size, -1).contiguous()
                            c0 = model[f'member_{idx+1}'].initial_c.expand(-1, batch_size, -1).contiguous()

                            if model[f'member_{idx+1}'].ext_vars:
                                inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                                exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                                targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                            else:
                                inps, exts, targets = batch_inps, None, batch_outs

                            out = model[f'member_{idx+1}'].forward(input_sequence = inps, 
                                                                   initial_states  =(h0,c0),
                                                                     batch_of_externals=exts)
                            loss = self.loss_fn(out.output_sequence, targets)
                            # total_loss += loss.item()
                            batch_count += 1

                            accumulated_loss += loss
                    
                        total_loss= accumulated_loss/ batch_count
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()


                        batch_losses.append(total_loss.item())



                        model[f'member_{idx+1}'].eval()
                        if model[f'member_{idx+1}'].ext_vars is not None:


                                
                                with torch.no_grad():

                                    out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                        calibration_window=calibration_window[:,:-model[f'member_{idx+1}'].ext_vars] , 
                                        n_steps=val_data.shape[0],
                                        externals_for_calibration=calibration_window[:,-model[f'member_{idx+1}'].ext_vars:] ,
                                        externals_for_forecasts=val_data[:,-model[f'member_{idx+1}'].ext_vars:])
                                    
                                    current_val_loss = self.loss_fn(forecasts, val_data[:,:-model[f'member_{idx+1}'].ext_vars]).item()
                                    
                        else:

                            with torch.no_grad():

                                out_seq, forecasts = model[f'member_{idx+1}'].forecast_seq_to_seq(
                                    calibration_window=calibration_window, n_steps=val_data.shape[0])
                                
                                
                                current_val_loss = self.loss_fn(forecasts, val_data).item()

                        forecasts_dict[f'member_{idx+1}'] = forecasts
                        member_val_losses.append(current_val_loss)
                        member_loss_summary[f'member_{idx+1}'] = {'train_loss':total_loss.item(), 'val_loss': current_val_loss}
                    
                    if float(np.median(member_val_losses)) < float(np.median(best_losses)):
                        
                        for idx_, model in enumerate(self.ensembles):

                            model[f'member_{idx_+1}'].save_checkpoint(epoch+1, member_val_losses[idx_],
                                                                    model[f'optimizer_{idx_+1}'],
                                                                    checkpoint_dir=self.save_dir,
                                                                    cleanup=True,
                                                                    add_stuffs=f"_member_{idx_+1}")
                            
                            best_losses[idx_] = member_val_losses[idx_]

                            self.save_forecasts_to_csv(out_seq=out_seq, var_names=[f"var_{i+1}" for i in range(model[f'member_{idx_+1}'].output_size)], 
                                                                forecast=forecasts_dict[f'member_{idx_+1}'], filename=f"forecasts_epoch_{epoch+1}_member_{idx_+1}_.csv")
                            print (f"✅ [Model {idx_+1}] New best val loss: {member_val_losses[idx_]:.6f} | forecast results saved")
                        print(f"✅ All Ensemble members saved based on median loss: {np.median(member_val_losses):.6f}")
                    else:
                        print(f"ℹ️ Median Ensemble train loss {float(np.median(member_val_losses)):.6f} not better than best {float(np.median(best_losses)) :.6f}. Skipping save.")

                # epoch_loss_summary
                    epoch_losses.append({ f'epoch{epoch+1}' :member_loss_summary})

                    print(f"[End of Epoch {epoch+1} | Best Median loss across all models: {float(np.median(best_losses)):.6f}")


                    self.ensembles = self.load_ensemble_checkpoints(checkpoint_paths=self.save_dir , train_type = train_type)

                self.save_epochs_losses_to_json(epoch_losses=epoch_losses)



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

    def train_forecast(self, data_loader: DataLoader,
                       calibration_window: torch.Tensor,
                       val_data: torch.Tensor,
                       num_epochs: int = 10,
                       backprop_mode: Literal['per_batch', 'per_epoch'] = 'per_batch') -> List[List[float]]:
        """
        Trains ensemble and saves based on validation loss.

        Returns
        -------
        List of per-epoch training losses for each model.
        """
        all_losses = [[] for _ in range(len(self.ensembles))]
        best_losses = [float('inf')] * len(self.ensembles)
        best_epoch = -1

        for epoch in range(num_epochs):
            epoch_losses = []
            forecasts_for_epoch = []

            for idx, model in enumerate(self.ensembles):
                optimizer = model[f'optimizer_{idx+1}']
                model[f'member_{idx+1}'].train()

                total_loss, accumulated_loss, batch_count = 0.0, 0.0, 0

                for batch_inps, batch_outs in data_loader:
                    batch_size = batch_inps.size(0)
                    h0 = model[f'member_{idx+1}'].initial_hidden_state.expand(-1, batch_size, -1).contiguous()

                    if model[f'member_{idx+1}'].ext_vars:
                        inps = batch_inps[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                        exts = batch_inps[:, :, -model[f'member_{idx+1}'].ext_vars:]#.to(self.device)
                        targets = batch_outs[:, :, :-model[f'member_{idx+1}'].ext_vars]#.to(self.device)
                    else:
                        inps, exts, targets = batch_inps, None, batch_outs

                    out = model[f'member_{idx+1}'].forward(input_sequence =inps, 
                                                           initial_states=h0, batch_of_externals=exts)
                    loss = self.loss_fn(out.output_sequence, targets)
                    total_loss += loss.item()
                    batch_count += 1

                    if backprop_mode == 'per_batch':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        accumulated_loss += loss

                if backprop_mode == 'per_epoch':
                    optimizer.zero_grad()
                    accumulated_loss.backward()
                    optimizer.step()

                avg_loss = total_loss / batch_count
                all_losses[idx].append(avg_loss)

                # self._save_losses_json(self.ensembles[f'member_{idx+1}'], avg_loss, idx)

                # Validation forecast
                model[f'member_{idx+1}'].eval()
                with torch.no_grad():
                    if model[f'member_{idx+1}'].ext_vars:
                        out_seq, forecast = model[f'member_{idx+1}'].forecast_seq_to_seq(
                            calibration_window[:, :-model[f'member_{idx+1}'].ext_vars],
                            n_steps=val_data.shape[0],
                            externals_for_calibration=calibration_window[:, -model[f'member_{idx+1}'].ext_vars:],
                            externals_for_forecasts=val_data[:, -model[f'member_{idx+1}'].ext_vars:]
                        )
                        val_loss = self.loss_fn(forecast, val_data[:, :-model[f'member_{idx+1}'].ext_vars]).item()
                    else:
                        out_seq, forecast = model[f'member_{idx+1}'].forecast_seq_to_seq(calibration_window, n_steps=val_data.shape[0])
                        val_loss = self.loss_fn(forecast, val_data).item()

                    epoch_losses.append(val_loss)
                    forecasts_for_epoch.append((out_seq, forecast))

                print(f"[Epoch {epoch+1} | Model {idx+1}] Val Loss: {val_loss:.6f}")

                # Saving logic
                if self.best_on == 'individual':
                    for idx, loss in enumerate(epoch_losses):
                        if loss < best_losses[idx]:
                            model[f'member_{idx+1}'].save_checkpoint(epoch+1, loss, model[f'optimizer_{idx+1}'],
                                                                    checkpoint_dir=self.save_dir, cleanup=True)
                            self._save_forecasts_to_csv(*forecasts_for_epoch[idx], idx, epoch)
                            best_losses[idx] = loss
                            print(f"✅ [Model {idx+1}] New best val loss: {loss:.6f}")
        if self.best_on == 'median':
            median_loss = float(torch.tensor(epoch_losses).median())
            # median_loss = torch.median(torch.tensor(epoch_losses)).item()
            if median_loss < min(best_losses):
                for idx in range(self.ensemble.n_ensemble):
                    model[f'member_{idx+1}'].save_checkpoint(epoch+1, epoch_losses[idx], model[f'optimizer_{idx+1}'],
                                                                checkpoint_dir=self.save_dir, cleanup=True)
                    self._save_forecasts_to_csv(*forecasts_for_epoch[idx], idx, epoch)
                    best_losses[idx] = epoch_losses[idx]
                best_epoch = epoch + 1
                print(f"✅ Ensemble saved based on median val loss: {median_loss:.6f}")



        return all_losses



    def _save_losses_json(self, model, losses: List[float], model_idx: int):
        path = os.path.join(self.save_dir, f"losses_model{model_idx+1}_{model.name}.json")
        with open(path, 'w') as f:
            json.dump([{"epoch": i+1, "avg_loss": l} for i, l in enumerate(losses)], f, indent=2)


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
                checkpoint = torch.load(checkpoint_path_, map_location=self.ensembles[idx][f'member_{idx+1}']._get_default_device())
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
                checkpoint = torch.load(checkpoint_path_, map_location=self.ensembles[idx][f'member_{idx+1}']._get_default_device())
                self.ensembles[idx][f'member_{idx+1}'].load_state_dict(checkpoint["model_state_dict"])
                self.ensembles[idx][f'optimizer_{idx+1}'].load_state_dict(checkpoint["optimizer_state_dict"])

                print(f"✅ Loaded Model {idx+1} from {checkpoint_path_} | Epoch {checkpoint['epoch']} | Loss: {checkpoint['loss']:.6f}")

            


        return self.ensembles
    

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

