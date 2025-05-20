from typing import Dict , Tuple
from datetime import date
# from datetime import datetime
# from torch.utils.data import Dataset, DataLoader
import os , time , json
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy
# from utils_fcts import LogCoshLoss
# from data_manip import SlidingWindowDataset
# slided_traindataset = SlidingWindowDataset(tensor_traindata, window_training)

# ct , fc = slided_traindataset.random_context_test_data_generator(tensor_traindata , 500, 25,  'random_')
# ct.shape , fc.shape

# ct.shape , fc.shape


 
class RNNs_modelling_scenario():
    folder_pre_trained = os.makedirs("results/pretrained_models/",  exist_ok=True)
    folder_pre_tested = os.makedirs("results/pretested_models/", exist_ok=True)
    folder_output_dicts = os.makedirs("results/output_dicts/",  exist_ok=True)
    folder_output_csv= os.makedirs("results/csv_files/",  exist_ok=True)

    def __init__(self,  context_window_size: int, forecast_horizon:int, num_epochs:int, simulation_id:str):
        # self.model = model
        # self.train_loader = train_loader
        self.forecast_horizon = forecast_horizon
        self.context_window_size =context_window_size
        self.num_epochs = num_epochs
        # self.learning_rate =learning_rate
        self.simulation_id = simulation_id
    

        # if optim_algo.lower() == "adam":
        #     self.optimizer = torch.optim.Adam(self.model.parameters() , lr = self.learning_rate)
        # elif  optim_algo.lower() == "sgd":
        #     self.optimizer = torch.optim.SGD(self.model.parameters() , lr = self.learning_rate)
        # else: 
        #     raise ValueError("optimizer_algo must be either 'adam' or 'sgd' ")

        # if loss_fct.lower() == 'mse':
        #     self.criterion = torch.nn.MSELoss()
        
        # elif loss_fct.lower() == 'logcosh':
        #     self.criterion =  LogCoshLoss()
        # else: 
        #     raise ValueError("loss_fct must be either 'mse' or 'logcosh' ")

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def model_name( self , custom_model  )-> str:

        """The model naming convention is 
        modeltype:_modelname_hd:_hidden_vars_ctxt:_ctxt_window_fc:_forecasthor_date_hour_
        """

        model_type = custom_model.__class__.__name__
        hidden_vars = custom_model.hidden_size
        #ctext_window =  self.context_window
        fcast_hor = self.forecast_horizon
        #today = date.today()
        # print(today.strftime("%d_%m"))

        model_full_name = f"{self.simulation_id}:_{model_type}:_hd:{hidden_vars}{custom_model.train_s0}_ctxt:{self.context_window_size}_fc:{fcast_hor}steps::_{self.num_epochs}epochs_trained_on_{date.today().strftime('%d_%m')}"
        #_at_{datetime.now().strftime("%H_%M_")}
        self.set_seed(42)
        return model_full_name
    

    def results_saver(self, custom_model, results_dictionary ):
        results_dictionary_cp = {k: v for k, v in results_dictionary.items() if k not in  ["model_", "pred_testdata" , "true_testdata"]}

        output_dicts = "results/output_dicts/"
        output_csv = "results/csv_files/"

        with open(f"{output_dicts}{custom_model.__class__.__name__}_{self.simulation_id}.json", "w") as file:
            json.dump(results_dictionary_cp, file)

        pd.DataFrame(results_dictionary['pred_testdata'], columns=['pred_x', 'pred_y', 'pred_z']).to_csv(f"{output_csv}/{custom_model.__class__.__name__}_{self.simulation_id}_pred.csv" , index=False)
        pd.DataFrame(results_dictionary['true_testdata'], columns=['true_x', 'true_y', 'true_z']).to_csv(f"{output_csv}/{custom_model.__class__.__name__}_{self.simulation_id}_true.csv" , index=False)
        return print("Training results saved successfully")

    def test_only(self, custom_model,  context_window:torch.Tensor, forecast_horizon :int , ) -> torch.Tensor:

        """
        Inputs Shape:
        context_window: [seq_len, features]
        forecast_horizon: int
        
        Outputs Shape:
        preds: [forecast_horizon,features]
        """
        self.set_seed(42)
        custom_model.eval()

        #with torch.no_grad():

        h0_c0_tuple= custom_model.initial_hidden_state(  batch_size = 1) #h0 , c0
        if custom_model.__class__.__name__ == 'LSTMModel':
            preds = self.forecast_seq_to_seq_lstm(custom_model, context_window, h0_c0_tuple , forecast_horizon)
        elif custom_model.__class__.__name__ == 'RNNModel':
            
            preds = self.forecast_seq_to_seq(custom_model, context_window, h0_c0_tuple , forecast_horizon)
        
        return preds


    def traintest_savebest( self ,  custom_model:nn.Module, train_loader ,context_window:torch.Tensor, 
                           true_testdata:torch.Tensor, criterion :nn.Module , optimizer :nn.Module  ) -> Dict:
    
        """ The train function """
        best_test_loss = float('inf')
        result_dict={}
        train_start_time = time.time()
        training_losses = []
        self.set_seed(42)
        folder_pre_tested = "results/pretested_models/"
        for epoch in range (self.num_epochs):

            epoch_start_time = time.time()
            

            for id_, (inputs_seq, targets_seq,) in enumerate(train_loader):
                custom_model.train()

                optimizer.zero_grad()

                h0_c0_tuple= custom_model.initial_hidden_state( batch_size = inputs_seq.size(0)) #h0 , c0
                _, outputs_seq , hTcT = custom_model(inputs_seq, h0_c0_tuple) #h0_c0_tuple ==> (h0, c0) , hT_cT_tuple ==> (hT, cT)

                loss = criterion(outputs_seq, targets_seq)

                
                loss.backward()
                optimizer.step()
                training_losses.append(loss.item())
                
                # print(f"train_loss = {loss.item()}")
                custom_model.eval()

                with torch.no_grad():

                    pred_testdata = self.test_only(custom_model, context_window , self.forecast_horizon)

                test_loss =  criterion( pred_testdata, true_testdata ).item()
                # print(f"test_loss = {test_loss.item()}")
                if test_loss < best_test_loss:

                    # try:
                    #     os.remove(f"{folder_pre_tested}Te_{self.model_name(custom_model)}.pt", exist_ok=True)

                    # except:

                    #     pass
                    
                    # torch.save(deepcopy(custom_model.state_dict()) , f"{folder_pre_tested}Te_{self.model_name(custom_model)}.pt")
                    # result_dict.update({'best_Te_model at':epoch +1 , "best_Te_loss": test_loss.item(), "model_name:": self.model_name()})

                    result_dict.update({'best_epoch':epoch +1 , "best_Te_loss": test_loss, 
                                        "model_": custom_model.state_dict(),
                                        "pred_testdata": pred_testdata.detach().cpu().numpy()})

                    best_test_loss = test_loss

                    # result_dict.update({'best_model': torch.save(self.model , f"{folder_pre_tested}{"Te"}_{self.model_name()}.pth")})
                    
            
            avg_training_loss = sum(training_losses)/len(training_losses)
            epoch_end_time = time.time()
            epoch_duration = round(epoch_end_time - epoch_start_time , 2)
            result_dict[f'epoch_{epoch +1}'] = {'avg_training_loss': avg_training_loss, 'test_loss': test_loss,
                                                 'time_taken': epoch_duration}

            if (epoch+1) % 1 == 0:
                print(f'Epoch {epoch +1}/{self.num_epochs} completed || Time taken: {epoch_duration} seconds || best epoch : {result_dict["best_epoch"]}')
                print(f'current test loss : {round(test_loss, 4)}|| best test loss : {round(result_dict["best_Te_loss"], 4)}')

            else:
                pass
        
        train_end_time = time.time()
        total_time = round(train_end_time - train_start_time, 2)

        print(f'Training completed. Total time taken: {total_time} seconds')
        result_dict['Total_TrainingTime'] = total_time
        result_dict["model_name"]= self.model_name(custom_model)
        result_dict["true_testdata"]= true_testdata.detach().cpu().numpy()
        torch.save(deepcopy(result_dict["model_"]) , f"{folder_pre_tested}Te_{self.model_name(custom_model)}.pt")
        print("results dicts, json's and csv's created successfully: check the results folder ")
        self.results_saver(custom_model , result_dict)

        return result_dict
    



    def train_only( self  , custom_model, train_loader , criterion :nn.Module , optimizer :nn.Module ) -> Dict:
    
        """ The train function """
        best_tr_loss = float('inf')
        result_dict={}
        train_start_time = time.time()
        folder_pre_trained = "results/pretrained_models/"
        training_losses = []
        for epoch in range (self.num_epochs):

            epoch_start_time = time.time()
            custom_model.train()

            for id_, (inputs_seq, targets_seq,) in enumerate(train_loader):
                optimizer.zero_grad()
                h0_c0_tuple= custom_model.initial_hidden_state( batch_size = inputs_seq.size(0)) #h0 , c0
                _, outputs_seq , hTcT = custom_model(inputs_seq, h0_c0_tuple) #h0_c0_tuple ==> (h0, c0) , hT_cT_tuple ==> (hT, cT)

                train_loss = criterion(outputs_seq, targets_seq)

                
                train_loss.backward()
                optimizer.step()
                training_losses.append(train_loss.item())

                if train_loss.item() < best_tr_loss:
                    torch.save(custom_model.state_dict() , f"{folder_pre_trained}Tr_{self.model_name(custom_model)}.pth")
                    result_dict.update({'best_Tr_model at':epoch +1 , "model_name:": self.model_name(custom_model) , "best_Tr_loss": train_loss.item()})
                    best_tr_loss = train_loss.item()


                
            avg_training_loss = sum(training_losses)/len(training_losses)
            epoch_end_time = time.time()
            epoch_time = round(epoch_end_time - epoch_start_time , 2)
            # print(f'Epoch {epoch +1} completed. Time taken: {epoch_time} seconds')
            print(f'Epoch {epoch +1}/{self.num_epochs} completed || Time taken: {epoch_time} seconds || best epoch : {result_dict["best_Tr_model at"]}')
            print(f'current training loss : {round(train_loss.item(), 4)}|| best training loss : {round(result_dict["best_Tr_loss"], 4)}')

            result_dict[f'epoch_{epoch +1}'] = {'avg_training_loss': avg_training_loss, 'time_taken': epoch_time}

            


        train_end_time = time.time()
        total_time = round(train_end_time - train_start_time, 2)

        print(f'Training completed. Total time taken: {total_time} seconds')
        result_dict['Total_TrainingTime'] = total_time
        return result_dict


    def forecast_seq_to_seq_lstm(self, model, input_sequence: torch.Tensor, h0_c0_tuple: Tuple[torch.Tensor, torch.Tensor], n_steps: int) -> torch.Tensor:
        with torch.no_grad():
            input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
            state_seq, output_seq, (hT, cT) = model.forward(input_sequence, h0_c0_tuple)

            future_hstates = torch.empty(1, n_steps + 1, h0_c0_tuple[0].size(2), dtype=torch.float32).to(model.device)
            future_cstates = torch.empty(1, n_steps + 1, h0_c0_tuple[1].size(2), dtype=torch.float32).to(model.device)
            future_outputs = torch.empty(1, n_steps + 1, input_sequence.size(2), dtype=torch.float32).to(model.device)

            future_hstates[:, 0, :] = hT.clone()
            future_cstates[:, 0, :] = cT.clone()
            future_outputs[:, 0, :] = output_seq[:, -1, :]

            for t_step in range(n_steps):
                next_input = future_outputs[:, t_step, :].unsqueeze(1)
                _, output_step, (h_future_T, c_future_T) = model.forward(next_input, (future_hstates[:, t_step, :].unsqueeze(0), future_cstates[:, t_step, :].unsqueeze(0)))
                future_hstates[:, t_step + 1, :] = h_future_T.squeeze(0)
                future_cstates[:, t_step + 1, :] = c_future_T.squeeze(0)
                future_outputs[:, t_step + 1, :] = output_step.squeeze(1)

        return future_outputs[0, 1:]
    
    def forecast_seq_to_seq(self, model, input_sequence: torch.Tensor, initial_states_s0: torch.Tensor, n_steps: int) -> torch.Tensor:

        """Inputs shape: 
        input_sequence: [seq_len, features]
        initial_states_s0 :  [num_layers, batch_size=1, hidden_size]
        n_steps: len_fcast_horizon
        
        This function takes the input_sequence, compute the forward pass 
        until the present time step for both S_(t) and y_(t) and then
        forecast the future outputs y_(t+1), ...., y_(t+n_steps)
        
        Outputs shape:
        future_outputs: [len_fcast_horizon, features]
            """
        with torch.no_grad():
            input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
            state_seq, output_seq, hstate_at_T = model.forward(input_sequence, initial_states_s0)

            future_hstates = torch.empty(1, n_steps + 1, initial_states_s0.size(2), dtype=torch.float32).to(model.device)
            future_outputs = torch.empty(1, n_steps + 1, input_sequence.size(2), dtype=torch.float32).to(model.device)

            future_hstates[:, 0, :] = hstate_at_T.clone()
            future_outputs[:, 0, :] = output_seq[:, -1, :]

            for t_step in range(n_steps):
                next_input = future_outputs[:, t_step, :].unsqueeze(1)
                _, output_step, h_future_T = model.forward(next_input, future_hstates[:, t_step, :].unsqueeze(0))
                future_hstates[:, t_step + 1, :] = h_future_T.squeeze(0)
                future_outputs[:, t_step + 1, :] = output_step.squeeze(1)

        return future_outputs[0, 1:]
    



 
class RNNs_ClimateModelling_scenario():
    folder_pre_trained = os.makedirs("results_climate/pretrained_models/",  exist_ok=True)
    folder_pre_tested = os.makedirs("results_climate/pretested_models/", exist_ok=True)
    folder_output_dicts = os.makedirs("results_climate/output_dicts/",  exist_ok=True)
    folder_output_csv= os.makedirs("results_climate/csv_files/",  exist_ok=True)

    def __init__(self,  context_window_size: int, forecast_horizon:int, num_epochs:int, simulation_id:str):
        # self.model = model
        # self.train_loader = train_loader
        self.forecast_horizon = forecast_horizon
        self.context_window_size =context_window_size
        self.num_epochs = num_epochs
        # self.learning_rate =learning_rate
        self.simulation_id = simulation_id
    

        # if optim_algo.lower() == "adam":
        #     self.optimizer = torch.optim.Adam(self.model.parameters() , lr = self.learning_rate)
        # elif  optim_algo.lower() == "sgd":
        #     self.optimizer = torch.optim.SGD(self.model.parameters() , lr = self.learning_rate)
        # else: 
        #     raise ValueError("optimizer_algo must be either 'adam' or 'sgd' ")

        # if loss_fct.lower() == 'mse':
        #     self.criterion = torch.nn.MSELoss()
        
        # elif loss_fct.lower() == 'logcosh':
        #     self.criterion =  LogCoshLoss()
        # else: 
        #     raise ValueError("loss_fct must be either 'mse' or 'logcosh' ")

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def model_name( self , custom_model  )-> str:

        """The model naming convention is 
        modeltype:_modelname_hd:_hidden_vars_ctxt:_ctxt_window_fc:_forecasthor_date_hour_
        """

        model_type = custom_model.__class__.__name__
        hidden_vars = custom_model.hidden_size
        #ctext_window =  self.context_window
        fcast_hor = self.forecast_horizon
        #today = date.today()
        # print(today.strftime("%d_%m"))

        model_full_name = f"{self.simulation_id}:_{model_type}:_hd:{hidden_vars}{custom_model.train_s0}_ctxt:{self.context_window_size}_fc:{fcast_hor}steps::_{self.num_epochs}epochs_trained_on_{date.today().strftime('%d_%m')}"
        #_at_{datetime.now().strftime("%H_%M_")}
        self.set_seed(42)
        return model_full_name
    

    def results_saver(self, custom_model, results_dictionary ):
        results_dictionary_cp = {k: v for k, v in results_dictionary.items() if k not in  ["model_", "pred_testdata" , "true_testdata"]}

        output_dicts = "results_climate/output_dicts/"
        output_csv = "results_climate/csv_files/"

        with open(f"{output_dicts}{custom_model.__class__.__name__}_{self.simulation_id}.json", "w") as file:
            json.dump(results_dictionary_cp, file)

        pd.DataFrame(results_dictionary['pred_testdata'], columns=['pred_wd20x', 'pred_wd20y', 'pred_ws60x','pred_wd60y' ]).to_csv(f"{output_csv}/{custom_model.__class__.__name__}_{self.simulation_id}_pred.csv" , index=False)
        pd.DataFrame(results_dictionary['true_testdata'], columns=['true_wd20x', 'true_wd20y', 'true_ws60x','true_wd60y']).to_csv(f"{output_csv}/{custom_model.__class__.__name__}_{self.simulation_id}_true.csv" , index=False)
        return print("Training results saved successfully")


    def target_externals_separator( self,  true_testdata : torch.Tensor , n_outputs) -> Tuple[torch.Tensor]:
        if true_testdata.dim() == 3:
            actual_true_test_data = true_testdata[:, :n_outputs, :].squeeze(2)
            actual_true_externals = true_testdata[:,n_outputs:,:].squeeze(2).unsqueeze(1).unsqueeze(1)
        elif true_testdata.dim() ==2:
            true_testdata = true_testdata.unsqueeze(2)
            actual_true_test_data = true_testdata[:, :n_outputs, :].squeeze(2)
            actual_true_externals = true_testdata[:,n_outputs:,:].squeeze(2).unsqueeze(1).unsqueeze(1)

        return actual_true_test_data , actual_true_externals
    

    def test_only(self, custom_model,  context_window:torch.Tensor, forecast_horizon :int , tensor_externals) -> torch.Tensor:

        """
        Inputs Shape:
        context_window: [seq_len, features]
        forecast_horizon: int
        
        Outputs Shape:
        preds: [forecast_horizon,features]
        """
        self.set_seed(42)
        custom_model.eval()

        #with torch.no_grad():
        h0_c0_tuple= custom_model.initial_hidden_state(  batch_size = 1) #h0 , c0
        if custom_model.__class__.__name__ == 'LSTMModelClimate':
            preds = self.forecast_seq_to_seq_lstm(custom_model, context_window, h0_c0_tuple , forecast_horizon ,tensor_externals)
        elif custom_model.__class__.__name__ == 'RNNModelClimate':
            
            preds = self.forecast_seq_to_seq(custom_model, context_window, h0_c0_tuple , forecast_horizon , tensor_externals)
        
        return preds


    def traintest_savebest( self ,  custom_model:nn.Module, train_loader ,context_window:torch.Tensor, 
                           true_testdata:torch.Tensor, criterion :nn.Module , optimizer :nn.Module  ) -> Dict:
    
        """ The train function """
        best_test_loss = float('inf')
        result_dict={}
        train_start_time = time.time()
        training_losses = []
        self.set_seed(42)
        folder_pre_tested = "results_climate/pretested_models/"


        actual_true_testdata , tens_future_externals = self.target_externals_separator (true_testdata , custom_model.output_size)
        for epoch in range (self.num_epochs):

            epoch_start_time = time.time()
            

            for id_, (inputs_seq, targets_seq,) in enumerate(train_loader):
                custom_model.train()

                optimizer.zero_grad()

                h0_c0_tuple= custom_model.initial_hidden_state( batch_size = inputs_seq.size(0)) #h0 , c0
                _, outputs_seq , hTcT = custom_model(inputs_seq, h0_c0_tuple) #h0_c0_tuple ==> (h0, c0) , hT_cT_tuple ==> (hT, cT)

                loss = criterion(outputs_seq, targets_seq[:,:,:custom_model.output_size])

                
                loss.backward()
                optimizer.step()
                training_losses.append(loss.item())
                
                # print(f"train_loss = {loss.item()}")
                custom_model.eval()

                with torch.no_grad():

                    pred_testdata = self.test_only(custom_model, context_window , self.forecast_horizon , tens_future_externals)

                test_loss =  criterion( pred_testdata, actual_true_testdata ).item()
                # print(f"test_loss = {test_loss.item()}")
                if test_loss < best_test_loss:

                    # try:
                    #     os.remove(f"{folder_pre_tested}Te_{self.model_name(custom_model)}.pt", exist_ok=True)

                    # except:

                    #     pass
                    
                    # torch.save(deepcopy(custom_model.state_dict()) , f"{folder_pre_tested}Te_{self.model_name(custom_model)}.pt")
                    # result_dict.update({'best_Te_model at':epoch +1 , "best_Te_loss": test_loss.item(), "model_name:": self.model_name()})

                    result_dict.update({'best_epoch':epoch +1 , "best_Te_loss": test_loss, 
                                        "model_": custom_model.state_dict(),
                                        "pred_testdata": pred_testdata.detach().cpu().numpy()})

                    best_test_loss = test_loss

                    # result_dict.update({'best_model': torch.save(self.model , f"{folder_pre_tested}{"Te"}_{self.model_name()}.pth")})
                    
            
            avg_training_loss = sum(training_losses)/len(training_losses)
            epoch_end_time = time.time()
            epoch_duration = round(epoch_end_time - epoch_start_time , 2)
            result_dict[f'epoch_{epoch +1}'] = {'avg_training_loss': avg_training_loss, 'test_loss': test_loss,
                                                 'time_taken': epoch_duration}

            if (epoch+1) % 1 == 0:
                print(f'Epoch {epoch +1}/{self.num_epochs} completed || Time taken: {epoch_duration} seconds || best epoch : {result_dict["best_epoch"]}')
                print(f'current test loss : {round(test_loss, 4)}|| best test loss : {round(result_dict["best_Te_loss"], 4)}')

            else:
                pass
        
        train_end_time = time.time()
        total_time = round(train_end_time - train_start_time, 2)

        print(f'Training completed. Total time taken: {total_time} seconds')
        result_dict['Total_TrainingTime'] = total_time
        result_dict["model_name"]= self.model_name(custom_model)
        result_dict["true_testdata"]= actual_true_testdata.detach().cpu().numpy()
        torch.save(deepcopy(result_dict["model_"]) , f"{folder_pre_tested}Te_{self.model_name(custom_model)}.pt")
        print("results dicts, json's and csv's created successfully: check the results folder ")
        self.results_saver(custom_model , result_dict)

        return result_dict
    



    def train_only( self  , custom_model, train_loader) -> Dict:
    
        """ The train function """
        best_tr_loss = float('inf')
        result_dict={}
        train_start_time = time.time()
        folder_pre_trained = "results_climate/pretrained_models/"
        training_losses = []
        for epoch in range (self.num_epochs):

            epoch_start_time = time.time()
            custom_model.train()

            for id_, (inputs_seq, targets_seq,) in enumerate(train_loader):
                self.optimizer.zero_grad()
                h0_c0_tuple= self.model.initial_hidden_state( batch_size = inputs_seq.size(0)) #h0 , c0
                _, outputs_seq , hTcT = custom_model(inputs_seq, h0_c0_tuple) #h0_c0_tuple ==> (h0, c0) , hT_cT_tuple ==> (hT, cT)

                train_loss = self.criterion(outputs_seq, targets_seq)

                
                train_loss.backward()
                self.optimizer.step()
                training_losses.append(train_loss.item())

                if train_loss.item() < best_tr_loss:
                    torch.save(custom_model.state_dict() , f"{folder_pre_trained}Tr_{self.model_name(custom_model)}.pth")
                    result_dict.update({'best_Tr_model at':epoch +1 , "model_name:": self.model_name(custom_model)})
                    best_tr_loss = train_loss.item()


                
            avg_training_loss = sum(training_losses)/len(training_losses)
            epoch_end_time = time.time()
            epoch_time = round(epoch_end_time - epoch_start_time , 2)
            # print(f'Epoch {epoch +1} completed. Time taken: {epoch_time} seconds')
            print(f'Epoch {epoch +1} completed || Time taken: {epoch_time} seconds || best epoch : {result_dict["best_Tr_model at"]}')
            print(f'current training loss : {round(train_loss.item(), 4)}|| best training loss : {round(result_dict["best_Tr_loss"], 4)}')

            result_dict[f'epoch_{epoch +1}'] = {'avg_training_loss': avg_training_loss, 'time_taken': epoch_time}

            


        train_end_time = time.time()
        total_time = round(train_end_time - train_start_time, 2)

        print(f'Training completed. Total time taken: {total_time} seconds')
        result_dict['Total_TrainingTime'] = total_time


    def forecast_seq_to_seq_lstm(self, model, input_sequence: torch.Tensor, 
                                 h0_c0_tuple: Tuple[torch.Tensor, torch.Tensor], n_steps: int , tensor_ext_vars:torch.Tensor) -> torch.Tensor:
        

        """Inputs shape: 
        input_sequence: [seq_len, features]
        initial_states_s0 :  [num_layers, batch_size=1, hidden_size]
        n_steps: len_fcast_horizon
        tensor_exogene_vars : [n_steps , 1 , n_extvars]  : n_extvars = 2  which (sin , cos) of time index

        
        This function takes the input_sequence, compute the forward pass 
        until the present time step for both S_(t) and y_(t) and then
        forecast the future outputs y_(t+1), ...., y_(t+n_steps)
        
        Outputs shape:
        future_outputs: [len_fcast_horizon, features]
            """
        

        with torch.no_grad():
            input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
            state_seq, output_seq, (hT, cT) = model.forward(input_sequence, h0_c0_tuple)

            future_hstates = torch.empty(1, n_steps + 1, h0_c0_tuple[0].size(2), dtype=torch.float32).to(model.device)
            future_cstates = torch.empty(1, n_steps + 1, h0_c0_tuple[1].size(2), dtype=torch.float32).to(model.device)
            future_outputs = torch.empty(1, n_steps + 1, output_seq.size(2), dtype=torch.float32).to(model.device)

            future_hstates[:, 0, :] = hT.clone()
            future_cstates[:, 0, :] = cT.clone()
            future_outputs[:, 0, :] = output_seq[:, -1, :]

            for t_step , id_tens_extvars in zip(range(n_steps) , tensor_ext_vars):
                next_input = torch.cat( (future_outputs[:, t_step, :].unsqueeze(1) , id_tens_extvars) , dim = 2) 
                _, output_step, (h_future_T, c_future_T) = model.forward(next_input, (future_hstates[:, t_step, :].unsqueeze(0), future_cstates[:, t_step, :].unsqueeze(0)))
                future_hstates[:, t_step + 1, :] = h_future_T.squeeze(0)
                future_cstates[:, t_step + 1, :] = c_future_T.squeeze(0)
                future_outputs[:, t_step + 1, :] = output_step.squeeze(1)

        return future_outputs[0, 1:]
    

    def forecast_seq_to_seq( self, model, input_sequence: torch.Tensor, initial_states_s0: torch.Tensor,
                             n_steps: int , tensor_ext_vars:torch.Tensor) -> torch.Tensor:

        """Inputs shape: 
        input_sequence: [seq_len, features]
        initial_states_s0 :  [num_layers, batch_size=1, hidden_size]
        n_steps: len_fcast_horizon
        tensor_exogene_vars : [n_steps , 1 , n_extvars]  : n_extvars = 2  which (sin , cos) of time index

        
        This function takes the input_sequence, compute the forward pass 
        until the present time step for both S_(t) and y_(t) and then
        forecast the future outputs y_(t+1), ...., y_(t+n_steps)
        
        Outputs shape:
        future_outputs: [len_fcast_horizon, features]
            """
        with torch.no_grad():
            input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
            state_seq, output_seq, hstate_at_T = model.forward(input_sequence, initial_states_s0)

            future_hstates = torch.empty(1, n_steps + 1, initial_states_s0.size(2), dtype=torch.float32).to(model.device)
            future_outputs = torch.empty(1, n_steps + 1, output_seq.size(2), dtype=torch.float32).to(model.device) # input_sequence

            future_hstates[:, 0, :] = hstate_at_T.clone()
            future_outputs[:, 0, :] = output_seq[:, -1, :]

            for t_step , id_tens_extvars in  zip(range(n_steps) ,tensor_ext_vars) :

                # concatenating future_inp with brought-in externals  at each time step in the future.
                next_input = torch.cat( (future_outputs[:, t_step, :].unsqueeze(1) , id_tens_extvars) , dim = 2) 
                _, output_step, h_future_T = model.forward(next_input, future_hstates[:, t_step, :].unsqueeze(0))
                future_hstates[:, t_step + 1, :] = h_future_T.squeeze(0)
                future_outputs[:, t_step + 1, :] = output_step.squeeze(1)

        return future_outputs[0, 1:]