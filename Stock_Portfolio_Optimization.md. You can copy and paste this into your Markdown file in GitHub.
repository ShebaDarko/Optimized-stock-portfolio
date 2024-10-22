# Stock Portfolio Optimization Project

## Overview
This project aims to build an optimized stock portfolio that maximizes returns given an initial budget of $1,000. Utilizing historical stock price data, we employ optimization techniques to allocate resources effectively across selected stocks.

## Problem Framing
- **Objective**: Construct a stock portfolio that maximizes returns within a $1,000 budget.
- **Constraints**: Adherence to budget and realistic stock market conditions, including price fluctuations and potential transaction costs.

## Assumptions
- **Market Efficiency**: Stock prices reflect all available information and follow a random walk.
- **Transaction Costs**: We assume no transaction fees to simplify calculations.
- **Risk Tolerance**: A moderate risk tolerance is adopted to balance high-return stocks with stable options.
- **Time Horizon**: The analysis is based on historical data, indicating a short- to medium-term investment strategy.

## Model Development

### Model Choice
- **Mean-Variance Optimization (MVO)**: Based on Modern Portfolio Theory (MPT), this method balances expected returns against risk. Key steps include:
  - Calculating expected returns and the covariance matrix of stock prices.
  - Using quadratic programming to find the optimal asset allocation.
- **Libraries Used**: 
  - **Pandas** for data manipulation
  - **NumPy** for numerical computations
  - **SciPy** or **CVXPY** for optimization

## API Creation
- **Framework**: Flask (or FastAPI) is employed to build a simple RESTful API.
- **Functionality**: The API accepts input parameters (e.g., stock symbols, historical data) and returns the optimized portfolio as a JSON response.
- **Endpoints**: 
  - `/optimize_portfolio`: Accepts budget and stock data as input and returns the allocation of funds across selected stocks.

# Trade-offs and Prioritization
- **Simplicity vs. Complexity**: Striking a balance between model sophistication and accessibility for understanding and implementation.
- **Performance vs. Interpretability**: Emphasizing a model that performs well while remaining easy to explain to non-technical stakeholders.


## Sample Code Snippet
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load stock price data
data = pd.read_csv('path_to_your_data.csv')
returns = data.pct_change().dropna()

# Define the optimization function
def portfolio_optimization(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -portfolio_return / portfolio_volatility  # Minimize negative return-to-risk ratio

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # weights must sum to 1
bounds = tuple((0, 1) for _ in range(len(data.columns)))  # weights must be between 0 and 1

# Run the optimization
mean_returns = returns.mean()
cov_matrix = returns.cov()
initial_weights = [1.0 / len(data.columns)] * len(data.columns)
optimized_results = minimize(portfolio_optimization, initial_weights, args=(mean_returns, cov_matrix),
                             method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = optimized_results.x
print("Optimal Portfolio Weights:", optimal_weights)
