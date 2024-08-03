from flask_restx import Resource, Namespace, fields
import numpy as np
import models
from datetime import datetime
import yfinance as yf
import pandas as pd

# borre joblib para probar pero quedo re desprolijo
# una vez que funcione la API:
"""
- hacer pruebas simples con la API (fechas invalidas y esas cosas)
- correr backtesting largo
- emprolijar notebooks y codigo
- completar diapos
"""
api = Namespace('predictive_model', description='Predictive model operations', path='/')
current_model = None
last_training = None

load_fields = api.model('PredictiveModel', {
    'path': fields.String(required=True, description='Path to load the model file')
})
predict_fields = api.model('PredictiveModel', {
    'date': fields.List(fields.Float, required=True, description='Date as yyyy-mm-dd')
})
training_fields = api.model('PredictiveModel', {
    'input_data': fields.List(fields.Float, required=True, description='Input data for prediction'),
    'labels': fields.List(fields.Float, required=True, description='Labels for training')
})

@api.route('/load')
class LoadModel(Resource):
    @api.expect(load_fields)
    def post(self):
        global current_model
        data = api.payload
        model_file = data['path']
        try:
            current_model = None # NOTE: path is relative to the os.cwd(), where the app was executed
        except FileNotFoundError:
            return {"message": "File not found"}, 400
        return {"message": "Model loaded successfully"}

import lightgbm as lgb
def train_model(data):
    lookahead = 1
    label = 'r1_fwd'
    features = data.columns.difference([label]).tolist()
    data = data.loc[pd.IndexSlice[:, :], features + [label]].dropna()
    categoricals = ['year', 'month', 'sector', 'weekday']
    for feature in categoricals:
        data[feature] = pd.factorize(data[feature], sort=True)[0]
    lgb_data = lgb.Dataset(data=data[features],
                       label=data[label],
                       categorical_feature=categoricals,
                       free_raw_data=False)
    def get_lgb_params(data, t=1, best=0):
        scope_params = ['lookahead', 'train_length', 'test_length']
        lgb_train_params = ['learning_rate', 'num_leaves', 'feature_fraction', 'min_data_in_leaf']
        param_cols = scope_params[1:] + lgb_train_params + ['boost_rounds']
        df = data[data.lookahead==t].sort_values('ic', ascending=False).iloc[best]
        return df.loc[param_cols]

    base_params = dict(boosting='gbdt', objective='regression', verbose=-1)
    lgb_daily_ic = pd.read_hdf('../notebooks/data/model_tuning.h5', 'lgb/daily_ic')
    models = []
    for position in range(7):
        params = get_lgb_params(lgb_daily_ic,
                                t=lookahead,
                                best=position)

        params = params.to_dict()

        for p in ['min_data_in_leaf', 'num_leaves']:
            params[p] = int(params[p])

        num_boost_round = int(params.pop('boost_rounds'))
        params.update(base_params)

        model = lgb.train(params=params,
                        train_set=lgb_data,
                        num_boost_round=num_boost_round)
        models.append(model)
    return models

TICKERS = ['MIRG.BA',
 'MOLA.BA',
 'TXAR.BA',
 'CGPA2.BA',
 'INTR.BA',
 'YPFD.BA',
 'SEMI.BA',
 'LONG.BA',
 'CADO.BA',
 'VALO.BA',
 'TRAN.BA',
 'HAVA.BA',
 'CTIO.BA',
 'METR.BA',
 'CEPU.BA',
 'BHIP.BA',
 'AUSO.BA',
 'LEDE.BA',
 'OEST.BA',
 'TECO2.BA',
 'MOLI.BA',
 'CELU.BA',
 'INVJ.BA',
 'POLL.BA',
 'ROSE.BA',
 'TGSU2.BA',
 'IRSA.BA',
 'DGCU2.BA',
 'SUPV.BA',
 'CARC.BA',
 'CAPX.BA',
 'PAMP.BA',
 'FERR.BA',
 'TGNO4.BA',
 'PATA.BA',
 'EDN.BA',
 'GCLA.BA',
 'MTR.BA',
 'SAMI.BA',
 'BMA.BA',
 'BPAT.BA',
 'GARO.BA',
 'MORI.BA',
 'BYMA.BA',
 'CECO2.BA',
 'GGAL.BA',
 'GRIM.BA',
 'GBAN.BA',
 'LOMA.BA',
 'DOME.BA',
 'DYCA.BA',
 'BBAR.BA',
 'ALUA.BA',
 'COME.BA',
 'FIPL.BA',
 'GAMI.BA',
 'BOLT.BA',
 'CRES.BA']

def get_data(end, size):
	n_tickers = 58 # hardcodeado

	DATA_STORE = '../notebooks/data/assets.h5'
	ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']
	with pd.HDFStore(DATA_STORE) as store:
		last_date = store['merval/prices'].tail(1).index[0][0]
		first_date = store['merval/prices'].head(1).index[0][0]

		# check if data is enough.
		end_timestamp = pd.Timestamp(end)
		if last_date < end_timestamp:
			# load recent days
			stock_data = yf.download(TICKERS, start=last_date, end=end, progress=False)
			# TODO: parsear el resultado de yfinance y cargarlo al final de store['merval/prices']
			end_timestamp = stock_data.index[-1]

		# load data
		prices = (store['merval/prices']
				.loc[pd.IndexSlice[:end, :], ohlcv]
				.tail(n=size*n_tickers)
				.rename(columns=lambda x: x.replace('adj_', ''))
				.swaplevel()
				.sort_index())
	return prices

import talib
from talib import RSI, BBANDS, MACD, ATR
def engineer_data(data):
    prices = data.sort_index()
    DATA_STORE = '../notebooks/data/assets.h5'
    with pd.HDFStore(DATA_STORE) as store:
        metadata = (store['merval/stocks'].loc[:, ['marketcap', 'sector']])

    prices.volume /= 1e3 # make vol figures a bit smaller
    prices.index.names = ['symbol', 'date']
    metadata.index.name = 'symbol'

    # RSI
    rsi = prices.groupby(level='symbol').close.apply(RSI)
    prices['rsi'] = rsi.values

    # BB
    def compute_bb(close):
        high, mid, low = BBANDS(close, timeperiod=20)
        return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)
    bb = prices.groupby(level='symbol').close.apply(compute_bb)
    prices['bb_high'] = bb['bb_high'].values
    prices['bb_low'] = bb['bb_low'].values
    prices['bb_high'] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
    prices['bb_low'] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)

    # NATR
    prices['NATR'] = prices.groupby(level='symbol', 
                                group_keys=False).apply(lambda x: 
                                                        talib.NATR(x.high, x.low, x.close))
    def compute_atr(stock_data):
        df = ATR(stock_data.high, stock_data.low, 
                stock_data.close, timeperiod=14)
        return df.sub(df.mean()).div(df.std())
    prices['ATR'] = (prices.groupby('symbol', group_keys=False)
                 .apply(compute_atr))
    
    # PPO
    by_ticker = prices.groupby('symbol', group_keys=False)
    prices['ppo'] = by_ticker.close.apply(talib.PPO)
    
    # MACD
    def compute_macd(close):
        macd = MACD(close)[0]
        return (macd - np.mean(macd))/np.std(macd)
    prices['MACD'] = (prices
                  .groupby('symbol', group_keys=False)
                  .close
                  .apply(compute_macd))
    
    # Combine price and metadata
    metadata.sector = pd.factorize(metadata.sector)[0].astype(int)
    prices = prices.join(metadata[['sector']])

    # Create dummy variables
    prices['year'] = prices.index.get_level_values('date').year
    prices['month'] = prices.index.get_level_values('date').month
    prices['weekday'] = prices.index.get_level_values('date').weekday

    # Compute forward returns (labels)
    by_sym = prices.groupby(level='symbol').close
    prices[f'r1'] = by_sym.pct_change(1)
    prices[f'r1_fwd'] = prices.groupby(level='symbol')[f'r1'].shift(-1)
    return prices

@api.route('/predict')
class Predict(Resource):
    @api.expect(predict_fields)
    def post(self):
        global current_model
        global last_training
        data = api.payload
        # Get data from the request body
        date = data['date']

        # Check if there is a prediction stored
        db_prediction = models.Prediction.query.filter_by(date=date).first()
        if db_prediction is not None:
            return {"prediction": db_prediction.prediction}

        size = 34

        # Check if retraining is needed
        training_needed = not current_model or abs((last_training - pd.to_datetime(date)).days) > 63
        if training_needed:
            size += 252
            last_training = pd.to_datetime(date)

        # get data from last 252 days
        data = get_data(date, size)

        # Engineer input data
        daily_df = engineer_data(data)

        # Train model if needed
        if training_needed:
            current_model = train_model(daily_df)

        # Predict with all models and take each mean
        today_df = daily_df.xs(key=date, level='date').drop(columns=['r1_fwd'])
        predictions = []
        for model in current_model:
            predictions.append(model.predict(today_df.loc[:, model.feature_name()]))
        prediction = sum(predictions) / len(predictions)
        today_df = today_df.assign(prediction=prediction)

        # Apply strategy
        top_predictions = today_df[today_df['prediction'] > 0].nlargest(5, 'prediction')
        open_positions = dict()
        if len(top_predictions) == 0:
            open_positions['cash'] = 1
        else:
            # update open_positions
            allocation = 1 / len(top_predictions)
            for ticker in top_predictions.index:
                open_positions[ticker] = allocation

        # Store result in database
        new_prediction = models.Prediction(date=date, prediction=open_positions)
        models.db.session.add(new_prediction)
        models.db.session.commit()

        return open_positions

@api.route('/partial_fit')
class Train(Resource):
    @api.expect(training_fields)
    def post(self):
        data = api.payload
        input_data = data['input_data']
        labels = data['labels']
        if not current_model:
            return {"message": "No model loaded"}, 400
        # TODO: validate if the loaded model accepts partial fitting
        current_model.partial_fit(input_data, labels)
        return {"message": "Training completed successfully"}

@api.route('/stats')
class Stats(Resource):
    @api.expect()
    def get(self):
        global current_model

        historical_predictions = models.Prediction.query.all()

        sorted_predictions = sorted(historical_predictions, key=lambda x: datetime.strptime(x.month, "%Y-%m"))
        if len(sorted_predictions) <= 1:
            return {'index': [], 'portfolio': [], 'dates': []}
        dates = [datetime.strptime(pred.month, "%Y-%m") for pred in sorted_predictions]
        
        unique_tickers = set()
        unique_tickers.add("SPY") # FIX: indice merval no tiene todos los datos en yfinance. Sacarlos de otra api. Por ahora, se usa SP500
        for pred in sorted_predictions:
            for (ticker, _) in pred.prediction:
                unique_tickers.add(ticker) 

        def get_next_month(date_str):
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date.month == 12:
                next_month = date.replace(year=date.year + 1, month=1, day=1)
            else:
                next_month = date.replace(month=date.month + 1, day=1)
            return next_month.strftime("%Y-%m-%d")

        prices = yf.download(unique_tickers, start=sorted_predictions[0].month+'-01', end=get_next_month(sorted_predictions[-1].month+'-01'), progress=False, interval='1mo')
        value_merval = [prices.loc[prices.index == date, ('Close', 'SPY')].iloc[-1] for date in dates]
        value_predictions = [sum([((prices.loc[prices.index == datetime.strptime(pred.month, "%Y-%m"), ('Close', ticker)].iloc[-1]) * fraction) for (ticker, fraction) in pred.prediction]) for pred in sorted_predictions]

        return {'index_value': value_merval, 'portfolio_value': value_predictions, 'dates': [pred.month for pred in sorted_predictions]}