import datetime
import requests
import json
from tqdm import tqdm
import pandas as pd
import d2api
import time
from datetime import timedelta
import os
import numpy as np
import model_stuf
from gensim.models import KeyedVectors

def create_draft_data_set_in_loop(matches,game_type):
    folder = game_type
    date = time.strftime("%Y-%m-%d")
    try:
        df = pd.read_csv(f'{folder}/hero_data_{date}.csv')
        r1 = list(df['r1'])
        r2 = list(df['r2'])
        r3 = list(df['r3'])
        r4 = list(df['r4'])
        r5 = list(df['r5'])
        d1 = list(df['d1'])
        d2 = list(df['d2'])
        d3 = list(df['d3'])
        d4 = list(df['d4'])
        d5 = list(df['d5'])
        radiant_win = list(df['radiant_win'])
        heroes = [r1, r2, r3, r4, r5, d1, d2, d3, d4, d5]
    except:
        df = pd.DataFrame()
        r1 = []
        r2 = []
        r3 = []
        r4 = []
        r5 = []
        d1 = []
        d2 = []
        d3 = []
        d4 = []
        d5 = []
        heroes = [r1, r2, r3, r4, r5, d1, d2, d3, d4, d5]
        radiant_win = []
    for match in matches:
        try:
            if len(match['result']['players'])<10:
                continue
        except:
            if len(match['players'])<10:
                continue
        draft_order = 0
        try:
            radiant_win.append(match['result']['radiant_win'])
            for player in match['result']['players']:
                heroes[draft_order].append(player['hero_id'])
                draft_order += 1
        except:
            radiant_win.append(match['radiant_win'])
            for player in match['players']:
                heroes[draft_order].append(player['hero_id'])
                draft_order += 1

    dff=pd.DataFrame()
    dff['r1'] = r1
    dff['r2'] = r2
    dff['r3'] = r3
    dff['r4'] = r4
    dff['r5'] = r5
    dff['d1'] = d1
    dff['d2'] = d2
    dff['d3'] = d3
    dff['d4'] = d4
    dff['d5'] = d5
    dff['radiant_win'] = radiant_win
    dff.to_csv(f'{folder}/hero_data_{date}.csv', index=False)


def updhelper(game_type,min_id = 99999999999,last_match_time=datetime.datetime.now()):
    td = timedelta(days = 30)
    api = d2api.APIWrapper(api_key='A0D129625665186D9C1E0BDBF6A0F7A9', parse_response=False)
    max_id = 0
    if game_type == 'ladder':
        response = requests.get('https://api.opendota.com/api/proPlayers')
    else:
        response = requests.get(f'https://api.opendota.com/api/proMatches?less_than_match_id={min_id}')
    print(response)
    dic = response.json()
    try:
        with open(f'last_id_{game_type}.txt', 'r', encoding='utf-8') as f:
            break_id = f.readline()
    except:
        break_id = '7500000000'
    break_id = int(break_id)

    if game_type == 'ladder':
        for i in tqdm(dic):
            print(i)
            try:
                last_match_time_ig = datetime.datetime.fromisoformat(i['last_match_time'][:-1])
                if last_match_time - last_match_time_ig < td:
                    Steam_id = i['account_id']
                    matches_json = requests.get(f'https://api.opendota.com/api/players/{Steam_id}/matches').json()
                    games = []
                    matches = matches_json
                    for match in matches:
                        id = int(match['match_id'])
                        if max_id < id:
                            max_id = id
                        if id < break_id:
                            break
                        if match['game_mode'] != 22:
                            continue
                        try:
                            match_details = api.get_match_details(id)
                            match_details = json.loads(match_details)
                            games.append(match_details)
                        except:
                            print('TIMEOUT')
                    create_draft_data_set_in_loop(matches=games,game_type=game_type)

            except:
                continue
    else:
        games = []
        for match in tqdm(dic):
            id = int(match['match_id'])
            if min_id>id:
                min_id = id
            if max_id < id:
                max_id = id
            if id < break_id:
                break
            try:
                match_details = api.get_match_details(id)
                match_details = json.loads(match_details)
                games.append(match_details)
            except:
                print('TIMEOUT')
        create_draft_data_set_in_loop(matches=games,game_type=game_type)
        if min_id>break_id:
            updhelper('progames', min_id=min_id)
    with open(f'last_id_{game_type}.txt', 'w', encoding='utf-8') as f:
        f.write(str(max_id))


def update_data():
    get_dat_dota_team_ratings()
    updhelper('ladder')
    updhelper('progames')
    update_wv(5)
    wv = KeyedVectors.load('word_vectors.kv')
    create_rivalry_matrix(wv)


def get_dat_dota_team_ratings():
    response = requests.get('https://api.datdota.com/api/ratings/regions')
    print(response)
    dic = response.json()
    pd.json_normalize(dic['data']).to_csv('pro_ratings.csv', index=False)
    df = pd.read_csv('pro_ratings.csv')
    df = df.drop(columns=['glickoRatingDate', 'eloRatingDate', 'winsLastMonth', 'lossesLastMonth', 'valveId', 'glicko.phi', 'glicko2.sigma'])
    return df


def get_Localized_hero_list(api = d2api.APIWrapper(api_key='A0D129625665186D9C1E0BDBF6A0F7A9', parse_response=False)):
    l = json.loads(api.get_heroes())
    Localized_hero_list = l['result']['heroes']
    #q = list(filter(lambda Localized_hero_list: Localized_hero_list['id'] == 1, Localized_hero_list))[0]['name'][14:]
    return Localized_hero_list

def read_data(include_today:bool):
    today_unix = time.time()
    today = time.strftime("%Y-%m-%d")
    old_ladder_data = pd.DataFrame(columns=['r1', 'r2', 'r3', 'r4', 'r5', 'd1', 'd2', 'd3', 'd4', 'd5', 'radiant_win'])
    old_pro_data = pd.DataFrame(columns=['r1', 'r2', 'r3', 'r4', 'r5', 'd1', 'd2', 'd3', 'd4', 'd5', 'radiant_win'])
    new_ladder_data = pd.read_csv(f'ladder/hero_data_{today}.csv')
    new_pro_data = pd.read_csv(f'progames/hero_data_{today}.csv')
    if include_today:
        mask = lambda x, _: x < 24 * 60 * 60 * 30
    else:
        mask = lambda x, y: x < 24 * 60 * 60 * 30 and y != f'hero_data_{today}.csv'

    for y in os.listdir('ladder/'):
        x = today_unix - os.path.getctime(f'ladder/{y}')
        if mask(x, y):
            oldy = pd.read_csv(f'ladder/{y}')  # ??
            old_ladder_data = pd.concat([oldy, old_ladder_data], join='inner')
    for y in os.listdir('progames/'):
        x = today_unix - os.path.getctime(f'ladder/{y}')
        if mask(x, y):  # last 10 days
            oldy = pd.read_csv(f'progames/{y}')
            old_pro_data = pd.concat([oldy, old_pro_data], join='inner')
    Bigus = pd.concat([old_ladder_data, old_pro_data], join='inner')
    return Bigus, new_ladder_data, new_pro_data, old_ladder_data, old_pro_data

def update_wv(vector_size):
    Bigus,_,_,_,_ = read_data(True)
    wv = model_stuf.w2v(Bigus, vector_size)
    wv.save('word_vectors.kv')

def get_data():
    wv = KeyedVectors.load('word_vectors.kv')
    _, new_ladder_data, new_pro_data, old_ladder_data, old_pro_data = read_data(False)

    today_ladder_X, today_ladder_Y = xy(new_ladder_data, wv)
    today_pro_X, today_pro_Y       = xy(new_pro_data, wv)
    old_ladder_X, old_ladder_Y     = xy(old_ladder_data, wv)
    old_pro_X, old_pro_Y           = xy(old_pro_data, wv)

    Big_X = pd.concat([old_ladder_X,today_ladder_X,old_pro_X,today_pro_X],join='inner',ignore_index=True)
    Big_Y = pd.concat([old_ladder_Y,today_ladder_Y,old_pro_Y,today_pro_Y],join='inner',ignore_index=True)

    #Big_X.reset_index(inplace=True)


    x ={'old_ladder':old_ladder_X, 'old_pro':old_pro_X, 'today_ladder':today_ladder_X,
        'today_pro':today_pro_X, 'Big': Big_X}
    y = {'old_ladder': old_ladder_Y, 'old_pro': old_pro_Y, 'today_ladder': today_ladder_Y,
          'today_pro': today_pro_Y, 'Big': Big_Y}
    return x,y

def xy(df, wv):
    df[:10] = df[:10].astype(int)
    try:
        df['radiant_win'] = df['radiant_win'].astype(bool)
        Y = df['radiant_win']
        X = df.drop(['radiant_win'], axis='columns')
    except:
        X = df
        Y=None
    cdf = pd.DataFrame(get_counter_df(df,wv))
    X = concat_hero2vec(word_vectors=wv, df=X)
    X = pd.concat([X,cdf],ignore_index=True, axis=1)
    return X, Y

def new_one_hot(word_vectors,df):
    one_hoted_list_radiant_team = np.zeros((len(df), len(word_vectors)*word_vectors.vector_size))
    one_hoted_list_dire_team = np.zeros((len(df), len(word_vectors)*word_vectors.vector_size))
    df = np.array(df)
    for j, herolist in enumerate(df):
        for i, heroid in enumerate(herolist):
            if i < 5:
                for v in range(word_vectors.vector_size):
                    one_hoted_list_radiant_team[j][word_vectors.key_to_index[heroid]*word_vectors.vector_size+v] = word_vectors[word_vectors.key_to_index[heroid]][v]
            else:
                for v in range(word_vectors.vector_size):
                    one_hoted_list_dire_team[j][word_vectors.key_to_index[heroid]*word_vectors.vector_size+v] = word_vectors[heroid][v]
    X = pd.DataFrame(np.concatenate((one_hoted_list_radiant_team, one_hoted_list_dire_team), axis=1))
    return X


def concat_hero2vec(word_vectors,df):
    df = np.array(df)
    arr = np.zeros((len(df),10*word_vectors.vector_size))
    for j, herolist in enumerate(df):
        q=[]
        for i, heroid in enumerate(herolist):
            q.append(word_vectors[word_vectors.key_to_index[heroid]])
        arr[j] = np.array(q).flatten()
    x = pd.DataFrame(arr)
    #print(x.info())
    return x


def get_log5(pA,pB):
    return (pA-pA*pB)/(pA+pB-2*pA*pB)


def create_rivalry_matrix(wv):
    #Big,_,_,_,_ = read_data(True)
    #Big.reset_index(inplace=True,drop=True)
    #Bigs = Big.iloc[:, :10].copy()
    hl = get_Localized_hero_list()
    hero_count = len(hl)
    arr = np.zeros((hero_count,hero_count))
    for hero_id in hl:
        total_matches = 0
        total_wins = 0
        hero_id_i = wv.key_to_index[hero_id['id']]
        #col_ser = wdf.eq(hero_id['id'],axis=0).idxmax(axis=1)
        #wins = Big.loc[col_ser.index].copy()
        #for i, heros in wdf.iterrows():
        #    if col_ser[i][0] == 'r':
        #        win = wins['radiant_win'][i]
        #    else:
        #        win = not wins['radiant_win'][i]
        #    for h in heros:
        #        arr_cw[hero_id_i][wv.key_to_index[h]] += float(win)
        #        arr_ct[hero_id_i][wv.key_to_index[h]] += 1.0
        #print(arr_cw[hero_id_i])
        #print(arr_ct[hero_id_i])
        #for hss in hl:
        #    hss_id_i = wv.key_to_index[hss['id']]
        #    #print(arr_cw[hero_id_i][hss_id_i])
        #    arr[hero_id_i][hss_id_i] = (arr_cw[hero_id_i][hss_id_i]/arr_ct[hero_id_i][hss_id_i])
        response = requests.get(f'https://api.opendota.com/api/heroes/{hero_id["id"]}/matchups')
        while response.status_code != 200:
            time.sleep(10)
            response = requests.get(f'https://api.opendota.com/api/heroes/{hero_id["id"]}/matchups')
        dic = response.json()
        for hero in dic:
            total_wins += float(hero['wins'])
            total_matches += float(hero['games_played'])
            arr[hero_id_i][wv.key_to_index[hero['hero_id']]] = float(hero['wins'])/float(hero['games_played'])
        arr[hero_id_i][hero_id_i] = total_wins/total_matches
    df = pd.DataFrame(arr)
    pd.DataFrame(arr).to_csv('rivalry_matrix.csv',index=False)

def get_counter_value(heroidlist_radiant, heroidlist_dire, wv):
    l=[]
    rm = pd.read_csv('rivalry_matrix.csv').to_numpy()
    for i in heroidlist_radiant:
        ll=[]
        for j in heroidlist_dire:
            rh = wv.key_to_index[i]
            dh = wv.key_to_index[j]
            real = rm[rh][dh]
            l5 = get_log5(rm[rh][rh], rm[dh][dh])
            ll.append(real-l5)
        l.append(np.average(np.array(ll)))
    return l

def get_counter_df(dff,wv):
    df = dff.reset_index(drop=True)
    arr = np.zeros((len(df),5))
    for i,v in df.iterrows():
        arr[i] = get_counter_value(v[:5].values,v[5:10].values,wv)
    return arr

