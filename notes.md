## Notes : Future Work

This outlines using RL to mimic/improve politician investment strategies, using EDGAR data, stock fundamentals, technical indicators, and deep learning.

### Data Acquisition and Preprocessing:

*   **EDGAR Data:** Scrape SEC Form 4 filings for politician stock transactions.
*   **Stock Fundamentals:** Gather stock data (revenue, earnings, etc.) from financial data providers.
*   **Technical Indicators:** Calculate technical indicators (moving averages, RSI, MACD).
*   **Data Cleaning & Feature Engineering:** Handle missing data; create features like transaction size, holding period, profit/loss.
*   **Data Storage:** Store processed data (CSV, Parquet, database).

### Individual Politician Models:

*   **Model Creation:** Develop models for each politician with sufficient data.
*   **Feature Sets:** Tailor features based on traded stocks and available data.
*   **Hyperparameter Tuning:** Optimize hyperparameters for each model.

### Reinforcement Learning with Stable Baselines 3:

*   **Environment Design:**
    *   **State Space:** Stock prices, technical indicators, fundamentals, holdings, time.
    *   **Action Space:** Buy, sell, or hold decisions.
    *   **Reward Function:** Incentivize profitable trades, penalize losses (Sharpe ratio, Sortino ratio).
*   **Algorithm Selection:** Experiment with PPO, A2C, DQN, SAC.
*   **Training and Evaluation:** Train on historical data, evaluate on out-of-sample data (cumulative returns, Sharpe ratio, drawdown).
*   **Modular Model Importing:** Implement modular system for importing RL models.

### Microsoft Qlib Adaptation and Integration:

*   **Qlib Integration:** Adapt RL environment/models for Qlib.
*   **Multi-Model Validation:** Use Qlib to validate multiple models.
*   **Trading Actions:** Execute trades based on RL agent decisions in Qlib.
*   **Portfolio Optimization:** Use Qlib to allocate capital across models/assets.

### Transformer Model Experimentation (StockFormer etc.):

*   **Model Selection:** Explore transformer models like StockFormer.
*   **Input Representation:** Adapt input for transformer architecture.
*   **Training and Fine-tuning:** Train/fine-tune models for predicting trading behavior.
*   **Integration with RL:** Integrate transformer models into RL agent.
*   **Attention Visualization:** Analyze attention weights for insights.

### Model Validation and Backtesting:

*   **Backtesting Framework:** Develop a backtesting framework.
*   **Transaction Costs:** Incorporate transaction costs.
*   **Risk Management:** Implement stop-loss orders and position sizing.
*   **Performance Metrics:** Evaluate using cumulative returns, Sharpe ratio, maximum drawdown, volatility, information ratio.
*   **Statistical Significance:** Assess statistical significance of results.

### Deployment and Monitoring:

*   **Real-time Data Feeds:** Integrate real-time data.
*   **Automated Trading System:** Develop an automated trading system.
*   **Performance Monitoring:** Monitor performance and retrain models.
*   **Regulatory Compliance:** Ensure compliance.
