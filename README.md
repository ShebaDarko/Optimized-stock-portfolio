# Coding Challenge: Portfolio Optimization

## 🚀 Overview

Welcome to the **Portfolio Optimization** project! This repository aims to develop an intelligent stock portfolio optimization model that maximizes returns while adhering to realistic constraints. The project is built around a dynamic approach to stock selection, capitalizing on daily stock price data to craft an optimized investment strategy.

## 🎯 Problem Statement

Building a portfolio of stocks that optimizes returns given an initial budget of **$1000**. The goal is to create a model that intelligently allocates this budget across a selection of stocks, taking into account their historical performance and risk factors.

### 📊 Data

Daily stock price data for a variety of stocks is provided to facilitate analysis and model training. You can download the dataset [here](https://drive.google.com/file/d/1vdougP5eBLb7geavZIt7QyXhLVq4ai4M/view?usp=sharing).

## 🛠️ Project Components

1. **Model Development**:
   - Create a robust model to optimize stock selection based on historical price data and return metrics.
   - Use libraries such as **Pandas**, **NumPy**, **SciPy**, and **CVXPY** for data manipulation and optimization.

2. **API Creation**:
   - Develop a simple **Flask API** that accepts user inputs (like stock selections and budget) and returns predictions for an optimized portfolio.
   - The API will provide easy integration for further applications and enhance user experience.

3. **Documentation**:
   - This repository includes well-documented code, detailed comments, and clear instructions for usage and setup.

## ⚙️ Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Pip (Python package manager)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ShebaDarko/Optimized-stock-portfolio.git
   cd Optimized-stock-portfolio


###  Running the API
To run the Flask API, execute the following command:
python src/api.py

## 📖 Usage Instructions

- Load the Jupyter notebook located in the `notebooks` directory for a detailed analysis of the stock portfolio optimization process.
- Utilize the functions in `src/portfolio_optimization.py` for core logic and calculations related to portfolio optimization.
- Use the API for quick predictions and to interface with your optimization logic programmatically.

## 🤝 Contribution

Contributions are welcome! Please feel free to fork the repository, create a new branch, and submit a pull request. Ensure to follow the project's coding guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Acknowledgements

- [Flask](https://flask.palletsprojects.com/) - for building the web API.
- [CVXPY](https://www.cvxpy.org/) - for convex optimization solutions.
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) - for data manipulation and analysis.

