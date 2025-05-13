import pandas as pd
import torch , os, csv
import matplotlib.pyplot as plt
# from .utils import  custom_fcts
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Optional, Literal
from glob import glob
from datetime import datetime
from collections import namedtuple


VanillaHCNNForwardOutput = namedtuple(
    "VanillaHCNNForwardOutput", 
    ["expectations", "states", "delta_terms", "forecasts", "future_states"]
)


# Named tuple for the output of the HCNN-pTF model
HCNNpTFForwardOutput = namedtuple(
    "HCNNpTFForwardOutput",
    ["expectations", "states", "delta_terms", "partial_delta_terms", "forecasts", "future_states"]
)

# Named tuple for the output of the HCNN-LForm model

HCNNLFormForwardOutput = namedtuple(
    "HCNNLFormForwardOutput",
    ["expectations", "states", "delta_terms", "forecasts", "future_states"]
)

# Named tuple for the output of the HCNN-LSpa model

HCNNLSpaForwardOutput = namedtuple(
    "SparseHCNNForwardOutput",
    ["expectations", "states", "delta_terms", "forecasts", "future_states"]
)



class VanillaHCNNEnsemble:
    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        n_ext_vars: Optional[int] = None,
        s0_nature: str = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        optimizer_type: Literal ["sgd", "adam"] = "adam",
        learning_rate: float = 1e-4):



        self.optimizer_type = optimizer_type
        self.n_ensemble = n_ensemble
        self.model_args = dict(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            n_ext_vars =  n_ext_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range
        )
        self.learning_rate= learning_rate

            
        # self.loss_fn = nn.MSELoss() if loss_fn == "mse" else nn.LogCoshLoss()
        
        
        
        # def _ensemble_instantiation(self):


        self.models = []
        for member in range(self.n_ensemble):
            model = Vanilla_Model(**self.model_args)
            model.name = f"ensemble_member_{member+1}:{model._generate_model_name()}"

            if self.optimizer_type == 'adam':
            
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            elif optimizer == 'sgd':
            
                optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

            self.models.append({f"member_{member+1}": model, 
                            f"optimizer_{member+1}": optimizer})
            


class HCNNpTFEnsemble:
    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        n_ext_vars: Optional[int] = None,
        s0_nature: str = 'random_',
        train_s0: bool = True,
        target_prob: float = 0.5,
        drop_output: bool = False,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        optinizer_type: Literal ["sgd", "adam"] = "adam",
        learning_rate: float = 1e-4):

        self.optimizer_type = optimizer_type
        self.n_ensemble = n_ensemble
        self.model_args = dict(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            n_ext_vars =  n_ext_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            target_prob=target_prob,
            drop_output=drop_output
            init_range=init_range
        )
        self.learning_rate= learning_rate

            
            
        self.models = []
        for member in range(self.n_ensemble):
            model = HCNNpTF_Model(**self.model_args)
            model.name = f"ensemble_member_{member+1}:{model._generate_model_name()}"

            if self.optimizer_type == 'adam':
            
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            elif optimizer == 'sgd':
            
                optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

            self.models.append({f"member_{member+1}": model, 
                            f"optimizer_{member+1}": optimizer})
            



class HCNNLFormEnsemble:
    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        n_ext_vars: Optional[int] = None,
        s0_nature: str = 'random_',
        train_s0: bool = True,
        init_range: Tuple[float, float] = (-0.5, 0.5),
        init_diag:float = 1.0,
        optimizer_type: Literal ["sgd", "adam"] = "adam",
        learning_rate: float = 1e-4):



        self.optimizer_type = optimizer_type
        self.n_ensemble = n_ensemble
        self.model_args = dict(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            n_ext_vars =  n_ext_vars,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range,
            init_diag = init_diag
        )
        self.learning_rate= learning_rate

            
        # self.loss_fn = nn.MSELoss() if loss_fn == "mse" else nn.LogCoshLoss()
        
        
        
        # def _ensemble_instantiation(self):


        self.models = []
        for member in range(self.n_ensemble):
            model = LForm_Model(**self.model_args)
            model.name = f"ensemble_member_{member+1}:{model._generate_model_name()}"

            if self.optimizer_type == 'adam':
            
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            elif optimizer == 'sgd':
            
                optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

            self.models.append({f"member_{member+1}": model, 
                            f"optimizer_{member+1}": optimizer})


class LSpaEnsemble:
    def __init__(
        self,
        n_ensemble: int,
        n_obs_vars: int,
        n_hid_vars: int,
        n_ext_vars: Optional[int] = None,
        s0_nature: str = 'random_',
        train_s0: bool = True,
        sparsity_ratio: float = 0.25,
        mask_type:  Literal["random_block", "non_obs_block"] = "non_obs_block",
        init_range: Tuple[float, float] = (-0.5, 0.5),
        optimizer_type: Literal ["sgd", "adam"] = "adam",
        learning_rate: float = 1e-4):



        self.optimizer_type = optimizer_type
        self.n_ensemble = n_ensemble
        self.model_args = dict(
            n_obs_vars=n_obs_vars,
            n_hid_vars=n_hid_vars,
            n_ext_vars =  n_ext_vars,
            sparsity_ratio=sparsity_ratio,
            mask_type=mask_type,
            s0_nature=s0_nature,
            train_s0=train_s0,
            init_range=init_range
        )
        self.learning_rate= learning_rate

            
        # self.loss_fn = nn.MSELoss() if loss_fn == "mse" else nn.LogCoshLoss()
        
        
        
        # def _ensemble_instantiation(self):


        self.models = []
        for member in range(self.n_ensemble):
            model = LSpa_Model(**self.model_args)
            model.name = f"ensemble_member_{member+1}:{model._generate_model_name()}"

            if self.optimizer_type == 'adam':
            
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            elif optimizer == 'sgd':
            
                optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

            self.models.append({f"member_{member+1}": model, 
                            f"optimizer_{member+1}": optimizer})
            
