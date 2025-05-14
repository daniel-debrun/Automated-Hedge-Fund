import torch
import pandas as pd

class ForecastingModel():
    def __init__(self, model_path: str, checkpoint_path: str, device: str = "cpu"):
        """
        Abstract forecasting model.

        Args:
            model_path (str): Path to the model file.
            checkpoint_path (str): Path to the checkpoint file.
            device (str): Device to use for training and inference. Defaults to "cpu".
        """
        self.device = device
        self.model = self.load_model(model_path)
        self.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Abstract method to load the model from a file.

        Args:
            model_path (str): Path to the model file.

        Returns:
            torch.nn.Module: The loaded model.
        """
        pass

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load the model's state dictionary from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint)

    def step(self, df: pd.DataFrame) -> list[float]:
        """
        Abstract method to make a prediction.

        Args:
            df (pd.DataFrame): DataFrame containing the input data.

        Returns:
            list[float]: The prediction.
        """
        pass