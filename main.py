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
import cProfile
import pstats
from tests import *



def main():
    #update_data()
    #update_wv(50)

    #test_remove_least_frequent()
    #print("Test passed!")
    #wv = KeyedVectors.load('word_vectors.kv')
    #qwe(wv)
    #print(wv.key_to_index)
    #update_wv(5)
    update_model()
    #start_tg()


cProfile.run('main()', 'profile_stats')

p = pstats.Stats('profile_stats')

p.sort_stats('cumulative').print_stats(10)




