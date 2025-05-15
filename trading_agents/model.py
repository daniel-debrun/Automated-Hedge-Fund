import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import List, Optional, Type, Dict, Any
import numpy as np
from agents.portfolio_manager import *

from abc import abstractmethod


class Agent(object):
    def __init__(self):
        pass
    
    @abstractmethod
    def trade(self, state):
        pass
    
    def train(self):
        pass
    
    @abstractmethod
    def load_model(self, model_path):
        pass
    
    @abstractmethod
    def save_model(self, model_path):
        pass

class TradingAgent(Agent):
    def __init__(
        self,
        tradable_universe: List[str],  # List of stock tickers available for trading.
        portfolio_agent: portfolio_management_agent,
        edgar_data: Dict[str, Any],      # Dictionary mapping each ticker to its EDGAR features.
        sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
        model_path: Optional[str] = None,
        policy: str = "MlpPolicy",
        verbose: int = 0,
        **model_kwargs,
    ):
        """
        tradable_universe: List of stock tickers.
        edgar_data: Dictionary mapping each ticker to its EDGAR features.
        """
        super().__init__()
        self.tradable_universe = tradable_universe
        self.portfolio_agent = portfolio_agent # TODO: Implement portfolio management agent with AgentState
        self.edgar_data = edgar_data
        self.sb3_class = sb3_class
        self.model_path = model_path
        self.policy = policy
        self.verbose = verbose
        self.model_kwargs = model_kwargs

        # Build custom trading environment for the given stocks and data.
        self.env = self._build_trading_env()
        self._initialize_model()

    def _build_trading_env(self):
        # TODO
        pass

    def _initialize_model(self):
        if self.model_path is None:
            self.model = self.sb3_class(
                self.policy,
                self.env,
                verbose=self.verbose,
                **self.model_kwargs,
            )
        else:
            self.model = self.sb3_class.load(self.model_path)
            self.model.set_env(self.env)

    def predict(self, obs) -> np.ndarray:
        """Implements Agent.predict to predict an action given an observation.
        Args:
            obs (np.ndarray): The observation from the environment.
        Returns:
            np.ndarray: The predicted action(s) of length = len(tradable_universe).
                        Each action is an array, index 0 is the trade action /e [0,1] for sell/buy,
                        index 1 is proportion of the portfolio to allocate to the trade.
        """
        action, next_state = self.model.predict(obs)
        return action

    def learn(self, total_timesteps: int, log_interval: int = 1) -> None:
        """Implements Agent.learn to train the model.
        Args:
            total_timesteps: The number of timesteps to train the model.
            log_interval: The interval at which to log training progress.
        """
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

    def trade(self, state):
        """Implements Agent.trade by executing a trade given a state.
        Args:
            state: The current state of the environment.
        Returns:
        """
        return self.predict(state)

    def load_model(self, model_path: str):
        """Implements Agent.load_model to load a model from disk.
        Args:
            model_path: The path to the model file.
        """
        if not model_path:
            raise ValueError("Model path cannot be empty.")
        self.model = self.sb3_class.load(model_path)
        self.model.set_env(self.env)
        self.model_path = model_path

    def save_model(self, model_path: str):
        """Implements Agent.save_model to save the current model to disk.
        Args:
            model_path: The path to the model file.
        """
        self.model.save(model_path, include=['num_timesteps'])