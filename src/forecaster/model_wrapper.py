import torch
import pandas as pd

"""
Example folder structure:

Automated-Hedge-Fund/
├── src/
│   ├── forecaster/
│   │   ├── model_wrapper.py
│   │   └── ...
│   └── stockformer/
│       ├── __init__.py
│       ├── model.py

"""
from stockformer.model import StockFormer
# ...

class ForecastingModel():
    """A forecasting model wrapper providing a common API for different models.

    This class connects the main program to the unique preprocessing and inference
    logic of a given model, allowing interchangeable use of various forecasting models
    through a unified interface.
    """
    def __init__(self, model_type: str, model_path: str, checkpoint_path: str, device: str = "cpu"):
        """
        Initialize the ForecastingModel.

        Args:
            model_path (str): Path to the model file.
            checkpoint_path (str): Path to the checkpoint file.
            device (str): Device to use for training and inference. Defaults to "cpu".
        """
        self.device = device
        self.lookback = 0
        model_type = eval(model_type)
        # Import and initialize the correct model class based on model_type
        if type(model_type) == StockFormer:
            self.model = StockFormer
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.load_model(model_path, checkpoint_path)
        self.model.eval()  # Set the model to evaluation mode

    def load_model(self, model_path: str, checkpoint_path: str):
        """
        Abstract method to load the model from a file.

        Args:
            model_path (str): Path to the model file.

        Returns:
            torch.nn.Module: The loaded model.
        """
        # Load model parameters from the saved checkpoint
        checkpoint = torch.load(model_path, weights_only=True)
        # Extract model configuration
        lag = checkpoint['lag']
        lead = checkpoint['lead']
        self.lookback = checkpoint['lookback'] # Minimum look-back period for standardization
        n_features = checkpoint['features']
        embed_dim = checkpoint['embed_dim']
        num_heads = checkpoint['num_heads']
        ff_dim = checkpoint['ff_dim']
        num_layers = checkpoint['num_layers']
        dropout = checkpoint['dropout']
        seq_len = lag + lead
        tickers = checkpoint['tickers']

        # Instantiate the Transformer model with the loaded parameters
        self.model = self.model(
            seq_len=seq_len,
            features=n_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

    def step(self, df: pd.DataFrame) -> list[float]:
        """
        Abstract method to make a prediction.

        Args:
            df (pd.DataFrame): DataFrame containing the input data (OHLCV)

        Returns:
            list[float]: The prediction.
        """
        pred = self.model.inference(df)
        return pred