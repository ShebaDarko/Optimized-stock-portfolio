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
 
    # Portfolio Optimization Explanation

The goal of portfolio optimization is to find the best allocation of investments across a set of assets (stocks, in this case) to maximize returns for a given level of risk, or alternatively, to minimize risk for a given level of expected return. The two key components involved in this process are:

1. **Expected Returns**: The anticipated return from each asset over a certain period.
2. **Risk (Volatility)**: The variability of returns, often measured by the standard deviation of returns or by the covariance of the asset returns.

## Mathematical Formulation

### 1. Expected Returns

Let:
- \( R_i \) be the expected return of stock \( i \).
- \( w_i \) be the weight (allocation) of stock \( i \) in the portfolio.
: $$ [ E(R_p) = \sum_{i=1}^{n} w_i \cdot R_i ] $$ $$

E(R_p) = \sum_{i=1}^{n} w_i \cdot R_i $$ Where:

( n ) is the total number of stocks in the portfolio.

### 2. Portfolio Risk (Volatility)

The risk of the portfolio can be described using the covariance matrix \( \Sigma \), which captures how the returns of the stocks move together. The portfolio variance \( \sigma^2_p \) is defined as:

\[
\sigma^2_p = \mathbf{w}^T \Sigma \mathbf{w}
\]

Where:
- \( \mathbf{w} \) is a vector of weights \( [w_1, w_2, \ldots, w_n]^T \).
- \( \Sigma \) is the covariance matrix of the asset returns.

The portfolio volatility \( \sigma_p \) is simply the square root of the variance:

\[
\sigma_p = \sqrt{\sigma^2_p} = \sqrt{\mathbf{w}^T \Sigma \mathbf{w}}
\]

Copy code
### 3. Optimization Problem

The optimization problem can be framed as maximizing the expected return for a given level of risk. This leads to the following formulation:

**Objective Function**: Maximize the return-to-risk ratio, which can be expressed as:

\[
\text{Maximize} \quad \frac{E(R_p)}{\sigma_p}
\]

**Subject to**:
1. The sum of the weights must equal 1 (full allocation of the budget):
   \[
   \sum_{i=1}^{n} w_i = 1
   \]
2. No short selling (weights must be non-negative):
   \[
   w_i \geq 0 \quad \text{for all } i
   \]

## Implementation in the Code

In the provided code, the optimization is performed using the following steps:

1. **Define the Objective Function**: The function `portfolio_optimization` is defined to compute the negative return-to-risk ratio. This involves the calculations for both expected returns and volatility based on the weights.

   ```python
   def portfolio_optimization(weights, mean_returns, cov_matrix):
       portfolio_return = np.dot(weights, mean_returns)  # E(R_p)
       portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # σ_p
       return -portfolio_return / portfolio_volatility  # Minimize -E(R_p)/σ_p

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
