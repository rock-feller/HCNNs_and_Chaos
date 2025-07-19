import torch
from typing import Tuple , Literal , Union
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from typing import Literal
import random

class ChaoticSystemPlotter:
    def __init__(self, data: Union[torch.Tensor, pd.DataFrame] , system_name: str = "chaotic_system"):
        """
        data: either a torch.Tensor of shape (n, 3) or a DataFrame with 3 columns.
        """
        if isinstance(data, torch.Tensor):
            assert data.ndim == 2 and data.shape[1] == 3, "Tensor must be of shape (n, 3)"
            self.data_type = 'tensor'
            self.data = data
            self.labels = [f"{system_name} : {var_} variable" for var_ in ["x" , "y", "z"] ]
            self.system_name = system_name
        elif isinstance(data, pd.DataFrame):
            assert data.shape[1] == 3, "DataFrame must have 3 columns"
            self.data_type = 'dataframe'
            self.data = data
            self.labels = list(data.columns)
        else:
            raise TypeError("Input must be a torch.Tensor or a pandas.DataFrame")

    def plot_all_trajectories(self , render: Literal["vertical", "horizontal"] = "vertical"):
        if render == "vertical":

            fig, axs = plt.subplots(3, 1, figsize=(25, 10))
            colors = [ 'blue' ,'red', 'green']
            for i, ax in enumerate(axs):
                y = self.data[:, i].numpy() if self.data_type == 'tensor' else self.data.iloc[:, i]
                ax.plot(y, color=colors[i], label=self.labels[i])
                ax.set_title(self.labels[i])
                # ax.grid(True)
                ax.legend()
            fig.tight_layout()
            plt.show()

        elif render == "horizontal":
            fig, axs = plt.subplots(1, 3, figsize=(25, 4))
            colors = [ 'blue' ,'red', 'green']
            for i, ax in enumerate(axs):
                y = self.data[:, i].numpy() if self.data_type == 'tensor' else self.data.iloc[:, i]
                ax.plot(y, color=colors[i], label=self.labels[i])
                ax.set_title(self.labels[i])
                # ax.grid(True)
                ax.legend()
            fig.tight_layout()
            plt.show()


    def plot_true_and_pred(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor, 
        render: Literal["vertical", "horizontal"] = "horizontal"
    ):
        assert pred.shape == true.shape and pred.shape[1] == 3, "Shapes must match and be (n,3)"

        colors = ['red', 'green', 'blue']
        labels = ['x', 'y', 'z']

        if render == "vertical":
            fig, axs = plt.subplots(3, 1, figsize=(25, 10))
        elif render == "horizontal":
            fig, axs = plt.subplots(1, 3, figsize=(25, 4))
        else:
            raise ValueError("render must be either 'vertical' or 'horizontal'")

        for i, ax in enumerate(axs):
            ax.plot(true[:, i].numpy(), color=colors[i], label=f"true_{labels[i]}", linewidth=2)
            ax.plot(pred[:, i].numpy(), color=colors[i], linestyle='dotted', label=f"pred_{labels[i]}")
            ax.set_title(f"True vs Predicted - {labels[i]}")
            ax.legend()
            # ax.grid(True)

        fig.tight_layout()
        plt.show()

    def plot_ensemble(
        self,
        ensemble: torch.Tensor,
        ground_truth: torch.Tensor,
        add_mean_or_median: Literal['mean', 'median', 'both', None] = None,
        render: Literal["vertical", "horizontal"] = "vertical"
    ):
        assert ensemble.ndim == 3 and ensemble.shape[2] == 3, "Ensemble must be (m, n, 3)"
        assert ground_truth.shape == ensemble.shape[1:], "Ground truth must be of shape (n,3)"
        
        colors = ['red', 'green', 'blue']
        labels = ['x', 'y', 'z']
        m, n, _ = ensemble.shape

        # Choose subplot layout based on render mode
        if render == "vertical":
            fig, axs = plt.subplots(3, 1, figsize=(25, 10))
        elif render == "horizontal":
            fig, axs = plt.subplots(1, 3, figsize=(25, 4))
        else:
            raise ValueError("render must be either 'vertical' or 'horizontal'")

        for i, ax in enumerate(axs):
            # Plot ensemble members
            for j in range(m):
                ax.plot(ensemble[j, :, i].numpy(), color='grey', linestyle='dotted', alpha=0.5)

            # Plot ground truth
            ax.plot(ground_truth[:, i].numpy(), color=colors[i], label='ground truth', linewidth=2)

            # Add mean/median overlays if requested
            if add_mean_or_median in ['mean', 'both']:
                mean_vals = ensemble[:, :, i].mean(dim=0)
                ax.plot(mean_vals.numpy(), color=colors[i], linestyle='--', label='ensemble mean')

                ax.set_title(f"Ensemble members + ensemble mean +  Ground Truth - {labels[i]} variable")
                # ax.grid(True)
                ax.legend()

            if add_mean_or_median in ['median', 'both']:
                median_vals = ensemble[:, :, i].median(dim=0).values
                ax.plot(median_vals.numpy(), color=colors[i], linestyle='-.', label='ensemble median')

                ax.set_title(f"Ensemble members + ensemble median +  Ground Truth -{labels[i]}")
                ax.grid(True)
                ax.legend()

        fig.tight_layout()
        plt.show()






class WindSpeedDirectionPlotter:
    def __init__(self, df_true: pd.DataFrame, df_pred: pd.DataFrame):
        """
        Initialize with two DataFrames: one for ground truth and one for predictions.
        Both must have the same datetime index (in 'hh:mm' format) and corresponding columns.
        """
        required_columns = {'date_time', 'WS20M', 'WD20M', 'WS60M', 'WD60M'}
        if not required_columns.issubset(df_true.columns) or not required_columns.issubset(df_pred.columns):
            raise ValueError(f"Both DataFrames must contain the columns: {required_columns}")

        self.df_true = df_true.copy()
        self.df_pred = df_pred.copy()

        # Ensure 'date_time' is parsed correctly
        self.df_true['date_time'] = pd.to_datetime(self.df_true['date_time'], format='%H:%M')
        self.df_pred['date_time'] = pd.to_datetime(self.df_pred['date_time'], format='%H:%M')

    def plot_wind_metrics(
        self, 
        time_interval: int = 2, 
        render: Literal["vertical", "horizontal"] = "vertical", 
        true_first: bool = True
    ):
        """
        Plot wind speed and direction at 20M and 60M from both true and predicted DataFrames.
        - time_interval: used to space out the x-axis labels.
        - render: 'vertical' or 'horizontal' layout.
        - true_first: whether to plot true values before predicted.
        """
        metrics = [
            ('WS20M', 'Wind Speed 20M'),
            ('WD20M', 'Wind Direction 20M'),
            ('WS60M', 'Wind Speed 60M'),
            ('WD60M', 'Wind Direction 60M')
        ]

        n_plots = len(metrics)

        if render == "vertical":
            fig, axs = plt.subplots(n_plots, 1, figsize=(25, 12))
        elif render == "horizontal":
            fig, axs = plt.subplots(1, n_plots, figsize=(25, 4))
        else:
            raise ValueError("render must be either 'vertical' or 'horizontal'")

        if n_plots == 1:
            axs = [axs]

        for ax, (col, title) in zip(axs, metrics):
            if true_first:
                ax.plot(self.df_true['date_time'], self.df_true[col], marker='o', label='True ' + col, linewidth=2)
                ax.plot(self.df_pred['date_time'], self.df_pred[col], marker='o', linestyle='dotted', label='Pred ' + col)
            else:
                ax.plot(self.df_pred['date_time'], self.df_pred[col], marker='o', linestyle='dotted', label='Pred ' + col)
                ax.plot(self.df_true['date_time'], self.df_true[col], marker='o', label='True ' + col, linewidth=2)

            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True)
            ax.set_xticks(self.df_true['date_time'][::time_interval])
            ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()
        plt.show()
