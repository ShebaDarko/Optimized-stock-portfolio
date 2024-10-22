from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    tickers = data.get('tickers', [])
    weights = {ticker: 1/len(tickers) for ticker in tickers}  # Dummy logic for weights
    return jsonify({'tickers': tickers, 'weights': weights})

if __name__ == '__main__':
    app.run(debug=True)

