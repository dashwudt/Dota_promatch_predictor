import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scrap import *
from utils import *
from data import *
from tg import *
from model_stuf import *
from gensim.models import KeyedVectors
import matplotlib.pyplot as pl
import shap


update_data()

X,Y = update_model()

current_model = xgb.Booster()    # init model
current_model.load_model('model.json')

gini_scores(model = current_model,X = X,Y = Y,model_name='model')

xv=X['Big']

explainer = shap.TreeExplainer(current_model)
shap_values = explainer.shap_values(xv)
shap.summary_plot(shap_values, xv)


start_tg()
