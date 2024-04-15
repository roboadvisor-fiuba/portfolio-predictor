def identity(x):
    return x

def workflow(features, predictor=identity, signal=identity, selector=identity, optimizer=identity):
    '''
    predictor: model that receives features and returns predictions for each asset
    signal: analizes the prediction to return a buy/sell signal
    selector: marks the best assets based on the signals
    optimizer: allocates the capital in the selected assets. Returns positions of the portfolio
    '''
    predictions = predictor(features)
    signals = {k: signal(v) for k, v in predictions.items()}
    selection = selector(signals)
    portfolio = optimizer(selection)
    return portfolio