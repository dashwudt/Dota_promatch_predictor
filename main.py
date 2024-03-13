import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from model_stuf import *
import cProfile
import pstats

def main():
    update_data()
    update_wv(150)
    update_model()



cProfile.run('main()', 'profile_stats')

p = pstats.Stats('profile_stats')

p.sort_stats('cumulative').print_stats(10)




