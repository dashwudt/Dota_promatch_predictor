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





#update_wv(150)

update_data()
#read_data(50)
#wv = KeyedVectors.load('word_vectors.kv')
#qwe(wv=wv)

#update_model()



#start_tg()


