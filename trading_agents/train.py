from src.backtester import Backtester, set_backtester
from edgar import *
from collections import deque
from trading_agents.model import TradingAgent
import os
from datetime import datetime

# Training template for TradingAgent

def get_sec_filings(current_date, ticker):
    filings = get_filings(2021, 4).filter(form=['10-K', '10-Q'],
                                    amendments=True,
                                    ticker=ticker,
                                    exchange='NASDAQ')
    filings_list = []
    for filing in filings:
        filing_info = {
            'accession_number': filing.accession_number,
            'form': filing.form,
            'filed': filing.filed,
            'company': filing.company,
            'ticker': filing.ticker,
            'cik': filing.cik,
            'url': filing.url,
            'report_date': getattr(filing, 'report_date', None)
        }
        # Try to get the full text of the primary document
        try:
            filing_info['document_text'] = filing.document().text()
        except Exception as e:
            filing_info['document_text'] = None
        filings_list.append(filing_info)
    filings = filings_list
    return filings

# Training and Inference for TradingAgent


def get_recent_checkpoint_path(base_path):
    try:
        files = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        if not files:
            return None
        files.sort(key=os.path.getmtime)
        return files[-1]
    except Exception as e:
        print(f"Error: {e}")
        return None

def train_trading_agent(agent, backtester, table_rows, performance_metrics, train_timesteps=250_000, save_path='checkpoints'):
    # Example training loop
    dates = backtester.prepare_backtest()
    for step in range(train_timesteps):
        current_date = next(dates.iter, None)
        if current_date is None:
            break
        table_rows.extend(backtester.step(current_date))
        filings = {}
        for ticker in backtester.tickers:
            filings[ticker] = get_sec_filings(current_date, ticker)
        agent.step(current_date, table_rows, performance_metrics, filings)
        if step % 10_000 == 0:
            checkpoint_file = os.path.join(save_path, f"trading_agent_{step}_steps.chkpt")
            agent.save(checkpoint_file)

def inference_trading_agent(agent, backtester, table_rows, performance_metrics, checkpoint_dir='checkpoints'):
    recent_checkpoint = get_recent_checkpoint_path(checkpoint_dir)
    if recent_checkpoint:
        agent.load(recent_checkpoint)
  
    for current_date in backtester.prepare_backtest():
        table_rows.extend(backtester.step(current_date))
        filings = {}
        for ticker in backtester.tickers:
            filings[ticker] = get_sec_filings(current_date, ticker)
        agent.step(current_date, table_rows, performance_metrics, filings)

if __name__ == "__main__":

    # Choose mode: 'train' or 'inference'
    mode = os.getenv("AGENT_MODE", "train").lower()

    # SEC identifier
    set_identity(os.getenv("EDGAR_IDENTITY"))  # TODO: Download in advance to avoid usage limitations
    trading_agent = TradingAgent()  # init
    backtester = set_backtester()

    table_rows = deque(maxlen=10) # Ex. 10 intervals for recurrent agents
    
    performance_metrics = {
        'sharpe_ratio': None,
        'sortino_ratio': None,
        'max_drawdown': None,
        'long_short_ratio': None,
        'gross_exposure': None,
        'net_exposure': None
    }

    if mode == "train":
        train_trading_agent(trading_agent, backtester, table_rows, performance_metrics)
    else:
        inference_trading_agent(trading_agent, backtester, table_rows, performance_metrics)

    performance_df = backtester.analyze_performance()