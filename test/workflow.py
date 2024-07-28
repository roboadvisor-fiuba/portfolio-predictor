def identity(x):
    return x

def workflow(features, predictor=identity, signal=identity, selector=identity, optimizer=identity):
    '''
    predictor: model that receives features and returns a dictionary with predictions for each asset
    signal: analyzes each prediction to return a buy/sell signal
    selector: marks the best assets
    optimizer: allocates the capital in the selected assets. Returns a dictionary with positions for each asset
    '''
    predictions = predictor(features)
    signals = {k: signal(v) for k, v in predictions.items()}
    selection = selector(signals)
    portfolio = optimizer(selection)
    return portfolio