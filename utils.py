import numpy as np
import pandas as pd

import model_stuf
from model_stuf import *
from gensim.models import KeyedVectors

def get_coef(df,cassiifier,rm):
    wv = KeyedVectors.load('word_vectors.kv')
    X = df
    X, _ = xy(X, wv, rm)
    X = xgb.DMatrix(X)
    hero_pred = cassiifier.predict(X)
    rp = (hero_pred[0])
    return rp
def kelly(bank, classifier, df, rad_team_name, dire_team_name, coeff, rm):
    wv = KeyedVectors.load('word_vectors.kv')
    X = df

    X,_ = xy(X,wv, rm)

    X = xgb.DMatrix(X)

    hero_pred = classifier.predict(X)
    rp_g = model_stuf.get_glicko_prob(rad_team_name,dire_team_name)
    rp = (hero_pred[0]+rp_g)/2
    print(hero_pred, rp)
    dp = (1 - rp)
    cr = (1 / rp)
    cd = (1 / dp)
    #print(rp)
    coefficient_radiant = round(float(coeff[0]),2)
    coefficient_dire    = round(float(coeff[1]),2)
    if cr< coefficient_radiant:
        output = (f'{rad_team_name} vs {dire_team_name}'
                  f'\nBet {round(0.33 * bank * ((coefficient_radiant * rp-1)/(coefficient_radiant-1)),2)} on {rad_team_name}                   '
                  f'\nBB: {coefficient_radiant} vs {coefficient_dire} '
                  f'\nMe: {round(cr,2)} {round(cd,2)}')
    else:
        output = (f'{rad_team_name} vs {dire_team_name}'
                  f'\nBet {round(0.33 * bank * ((coefficient_dire * dp - 1) / (coefficient_dire - 1)),2)} on {dire_team_name}                  '
                  f'\nBB: {coefficient_radiant} vs {coefficient_dire} '
                  f'\nMe: {round(cr,2)} vs {round(cd,2)}')

    return str(output)

def balance_for_hero_position(df,target):
    X=pd.DataFrame()
    new_df = df.copy()
    X_test = pd.DataFrame( )
    full=pd.DataFrame()
    heropos1 = [1,2,3,4,5]
    for i in heropos1:
        heropos2 = heropos1.copy()
        heropos2.remove(i)
        new_df['r1'] = df[f'r{i}']
        new_df['d1'] = df[f'd{i}']
        for j in heropos2:
            new_df['r2'] = df[f'r{j}']
            new_df['d2'] = df[f'd{j}']
            heropos3 = heropos2.copy()
            heropos3.remove(j)
            for w in heropos3:
                new_df['r3'] = df[f'r{w}']
                new_df['d3'] = df[f'd{w}']
                heropos4 = heropos3.copy( )
                heropos4.remove(w)
                for e in heropos4:
                    new_df['r4'] = df[f'r{e}']
                    new_df['d4'] = df[f'd{e}']
                    heropos5 = heropos4.copy( )
                    heropos5.remove(e)
                    for ss in heropos5:
                        new_df['r5'] = df[f'r{ss}']
                        new_df['d5'] = df[f'd{ss}']
                        new_df['radiant_win'] = target
                        X_test = new_df[int(len(new_df)*.95):]
                        qq = new_df[:int(len(new_df)*.95)]
                        full = pd.concat([full, new_df], ignore_index=True)
                        X=pd.concat([X,qq],ignore_index=True)
    Y_test = X_test['radiant_win']

    Y=X['radiant_win']
    X_test.drop(columns=['radiant_win'],inplace=True)
    full.drop(columns=['radiant_win'], inplace=True)
    X.drop(columns=['radiant_win'],inplace=True)

    return X, X_test, Y, Y_test, full

def heroes_onehot_encoder(df):#->encoded_df
    one_hoted_list_radiant_team = np.zeros((len(df),138))
    one_hoted_list_dire_team = np.zeros((len(df),138))
    df = np.array(df)
    for j,herolist in enumerate(df):
        for i,heroid in enumerate(herolist):
            if i<5:
                one_hoted_list_radiant_team[j][heroid-1] = 1
            else:
                one_hoted_list_dire_team[j][heroid-1] = 1
    return pd.DataFrame(np.concatenate((one_hoted_list_radiant_team, one_hoted_list_dire_team),axis=1))

