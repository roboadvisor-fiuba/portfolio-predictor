'''
Script to evaluate a model using zipline and pyfolio
'''

from zipline.api import order_target, record, symbol
from zipline.finance import commission, slippage
import pandas as pd
import pyfolio as pf


def initialize(context):
    context.asset = symbol('AAPL') # TODO: reemplazar por panel lider MERVAL
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0)) # TODO: ver comision MERVAL
    context.set_slippage(slippage.VolumeShareSlippage()) # TODO: ver slippage MERVAL


def handle_data(context, data):
    # TODO: usar modelo para predecir precio de stocks. Usar datos del episodio actual para predecir el valor en el siguiente. 
    # En base una regla de decision, comprar/vender stocks correspondientes.

    # order_target(ticker, 100) para comprar 100 acciones de un ticker
    # record(key=value) para almacenar datos en el dataframe de resultados
    pass


def analyze(context, perf):
    # TODO: revisar problemas de versionado entre zipline y pyfolio
    df = pd.DataFrame(perf)
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(df)
    pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions, round_trips=True)
