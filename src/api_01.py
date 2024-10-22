from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.optimize import minimize

app = Flask(__name__)

# Load the dataset (make sure to specify the correct file path)
file_path = "/home/bathsheba/Downloads/stock_sample_data.xlsx"  # Adjust this to your file location

@app.route('/optimize_portfolio', methods=['POST'])
def optimize_portfolio():
    # Load the data
    data = pd.read_csv(file_path, sep='\t')  # Use the appropriate separator

    # Process the data
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'context_id', 'day']
    data['open'] = pd.to_numeric(data['open'], errors='coerce')
    data['high'] = pd.to_numeric(data['high'], errors='coerce')
    data['low'] = pd.to_numeric(data['low'], errors='coerce')
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

    # Drop rows with NaN values
    data.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    # Calculate daily returns based on 'close' prices
    data['returns'] = data['close'].pct_change()  
    data.dropna(subset=['returns'], inplace=True)

    # Pivot the data to get stock returns in separate columns
    returns_df = data.pivot_table(values='returns', index='date', columns='context_id')
    returns_df = returns_df.dropna()  

    # Calculate expected returns and covariance matrix for all stocks
    expected_returns = returns_df.mean() * 252  
    cov_matrix = returns_df.cov() * 252  

    # Initial budget
    initial_budget = 1000

    # Portfolio optimization function
    def portfolio_optimization(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_risk  

    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  
    bounds = tuple((0, 1) for _ in range(len(expected_returns)))  

    # Initial guess (equal distribution)
    initial_weights = [1 / len(expected_returns)] * len(expected_returns)

    # Optimize
    result = minimize(portfolio_optimization, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Optimized weights
    optimized_weights = result.x

    # Calculate amount to invest in each stock
    investment_amounts = [weight * initial_budget for weight in optimized_weights]

    # Prepare the response
    stock_investments = pd.Series(investment_amounts, index=expected_returns.index).to_dict()
    
    return jsonify(stock_investments)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Specify the port here

