import math

from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning

import json
import pandas as pd
import numpy as np

from tools.api import get_price_data
from utils.progress import progress
from forecaster.model_wrapper import ForecastingModel

model_type = ''
model_path = ''
checkpoint_path = ''
device = ''

model = ForecastingModel(model_type=model_type, model_path=model_path, checkpoint_path=checkpoint_path, device=device)

# TODO: consider updating other agents with forecasted values

##### Forecasting Analyst #####
def forecasting_analyst_agent(state: AgentState):
    """
    Sophisticated forecasting analysis system.

    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize prediction for each ticker
    forecast_prediction = {}

    progress.update_status("forecasting_analyst_agent", ticker, "Fetching historical price data")

    # Adjust the lookback for the model specifications
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    time_delta = (end_dt - start_dt).days
    if time_delta > model.lookback:
        start_dt = end_dt - pd.Timedelta(days=model.lookback)
        start_date = start_dt.strftime("%Y-%m-%d")

    # Get the historical price data
    prices_df = get_price_data(
        tickers=tickers,
        start_date=start_date, # 30 day interval by default
        end_date=end_date,
    )
    if prices_df is None:
        progress.update_status("forecasting_analyst_agent", '', "Error: No price data found")
        return

    predictions = model.step(prices_df)
    # predictions = { ticker_idx : dict[preds], ...}
    for idx, ticker in enumerate(tickers):
        # Get the forecast for the current ticker
        # Perform any necessary post-processing on the predictions here...
        forecast_prediction[ticker] = predictions.get(idx, [])

    progress.update_status("forecasting_analyst_agent", '', "Done")

    # Create the technical analyst message
    message = HumanMessage(
        content=json.dumps(forecast_prediction),
        name="forecasting_analyst_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(forecast_prediction, "Forecasting prediction")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["forecasting_analyst_agent"] = forecast_prediction

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }

