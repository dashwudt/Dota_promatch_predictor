import datetime
from functools import lru_cache
import matplotlib.pyplot as plt
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
from sklearn.mixture import GaussianMixture



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


def create_draft_data_set(matches,game_type):
    folder = game_type
    date = time.strftime("%Y-%m-%d")
    try:
        df = pd.read_csv(f'new_data/{folder}/hero_data_{date}.csv')
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
        id = list(df['id'])
        radiant_win = list(df['radiant_win'])
        heroes = [r1, r2, r3, r4, r5, d1, d2, d3, d4, d5]
    except:
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
        id=[]
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
        id.append(match['id'])

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
    dff['id'] = id
    dff.to_csv(f'new_data/{folder}/hero_data_{date}.csv', index=False)


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
        with open(f'last_id_new_{game_type}.txt', 'r', encoding='utf-8') as f:
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
            if max_id <= id:
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
    with open(f'last_id_new_{game_type}.txt', 'w', encoding='utf-8') as f:
        f.write(str(max_id))


def update_data():
    get_dat_dota_team_ratings()
    create_new_dataset('ladder')
    create_new_dataset('progames')
    update_wv(5)
    create_rivalry_matrix()


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
    old_ladder_data = pd.DataFrame(columns=['r1', 'r2', 'r3', 'r4', 'r5', 'd1', 'd2', 'd3', 'd4', 'd5', 'radiant_win','id'])
    old_pro_data = pd.DataFrame(columns=['r1', 'r2', 'r3', 'r4', 'r5', 'd1', 'd2', 'd3', 'd4', 'd5', 'radiant_win','id'])
    try:
        new_ladder_data = pd.read_csv(f'new_data/ladder/hero_data_{today}.csv')
    except:
        new_ladder_data=pd.DataFrame()
    try:
        new_pro_data = pd.read_csv(f'new_data/progames/hero_data_{today}.csv')
    except:
        new_pro_data=pd.DataFrame()

    if include_today:
        mask = lambda x, _: x < 24 * 60 * 60 * 30
    else:
        mask = lambda x, y: x < 24 * 60 * 60 * 30 and y != f'hero_data_{today}.csv'

    for y in os.listdir('new_data/ladder/'):
        x = today_unix - os.path.getctime(f'new_data/ladder/{y}')
        if mask(x, y):
            oldy = pd.read_csv(f'new_data/ladder/{y}')  # ??
            old_ladder_data = pd.concat([oldy, old_ladder_data], join='inner')
    for y in os.listdir('new_data/progames/'):
        x = today_unix - os.path.getctime(f'new_data/progames/{y}')
        if mask(x, y):  # last 10 days
            oldy = pd.read_csv(f'new_data/progames/{y}')
            old_pro_data = pd.concat([oldy, old_pro_data], join='inner')

    Bigus = pd.concat([old_ladder_data, old_pro_data], join='inner')
    old_pro_data = old_pro_data.drop_duplicates(ignore_index=True)
    #old_pro_data = remove_rows_with_least_frequent_values(old_pro_data,10)
    old_pro_data = old_pro_data.reset_index(drop=True)

    return Bigus, new_ladder_data, new_pro_data, old_ladder_data, old_pro_data

def update_wv(vector_size):
    _,_,_,_,old_ladder_data = read_data(True)
    wv = model_stuf.w2v(old_ladder_data, vector_size)
    wv.save('word_vectors.kv')

def get_data():
    wv = KeyedVectors.load('word_vectors.kv')
    rm = pd.read_csv('rivalry_matrix.csv').to_numpy()
    Bigus, new_ladder_data, new_pro_data, old_ladder_data, old_pro_data = read_data(False)

    today_ladder_X, today_ladder_Y = xy(new_ladder_data, wv, rm)
    today_pro_X, today_pro_Y       = xy(new_pro_data, wv   , rm)
    old_ladder_X, old_ladder_Y     = xy(old_ladder_data, wv, rm)
    old_pro_X, old_pro_Y           = xy(old_pro_data, wv   , rm)
    Big_X, Big_Y =                   xy(Bigus,wv,rm)



    x ={'old_ladder':old_ladder_X, 'old_pro':old_pro_X, 'today_ladder':today_ladder_X,
        'today_pro':today_pro_X, 'Big': Big_X}
    y = {'old_ladder': old_ladder_Y, 'old_pro': old_pro_Y, 'today_ladder': today_ladder_Y,
          'today_pro': today_pro_Y, 'Big': Big_Y}
    return x,y

def xy(df, wv,rm):
    try:
        df['radiant_win'] = df['radiant_win'].astype(bool)
        Y = df['radiant_win']
        X = df.drop(['radiant_win','id'], axis='columns')
    except:
        X = df
        X = X.drop(['id'], axis='columns')
        Y=None
    #X = role_placement_model(X,wv)
    cdf = pd.DataFrame(get_counter_df(rm,X))
    X = concat_hero2vec(word_vectors=wv, df=X)
    X = pd.concat([X,cdf],ignore_index=True, axis=1)
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    #X = cdf
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


def concat_hero2vec(word_vectors: KeyedVectors, df: pd.DataFrame):
    # Preallocate the array with the correct shape
    arr = np.zeros((len(df), 10 * word_vectors.vector_size))
    # Iterate over the DataFrame rows
    for j, herolist in enumerate(df.values):
        # Preallocate a list to hold the flattened word vectors
        q = np.zeros((10, word_vectors.vector_size))
        # Iterate over the hero ids and collect their vectors
        for i, heroid in enumerate(herolist):
            q[i] = word_vectors[word_vectors.key_to_index[heroid]]
        # Flatten the collected vectors and assign to the corresponding row in the array
        arr[j] = q.flatten()
    # Convert the array to a DataFrame
    x = pd.DataFrame(arr)
    return x


@lru_cache(maxsize=None)
def get_log5(pA,pB):
    return (pA-pA*pB)/(pA+pB-2*pA*pB)



def get_counter_value(rivalry_matrix,heroidlist_radiant, heroidlist_dire):
    radiant_indices = [int(hero_id) for hero_id in heroidlist_radiant]
    dire_indices = [int(hero_id) for hero_id in heroidlist_dire]
    l5_values = np.array([get_log5(rivalry_matrix[idx,idx],rivalry_matrix[idy,idy]) for idx,idy in zip(radiant_indices,dire_indices)])
    real_values = rivalry_matrix[np.ix_(radiant_indices, dire_indices)]
    #counter_values = real_values - l5_values[:, None]
    counter_values = real_values / l5_values[:, None]
    #counter_values = real_values
    return np.mean(counter_values, axis=1)

def get_counter_df(rm,dff):
    df = dff.reset_index(drop=True)
    arr = np.zeros((len(df), 5))
    for i, row in enumerate(df.to_numpy()):
        arr[i] = get_counter_value(rm,row[:5], row[5:10])
    return arr

def fetch_matches(game_type, min_id):
    if game_type == 'ladder':
        response = requests.get('https://api.opendota.com/api/proPlayers')
    else:
        response = requests.get(f'https://api.opendota.com/api/proMatches?less_than_match_id={min_id}')

    if not response.ok:
        # If the response is not OK, raise an exception or handle it appropriately
        raise Exception(f"Error fetching data: {response.status_code}")

    return response.json()


def create_new_dataset(game_type, min_id=99999999999, last_match_time=datetime.datetime.now(),max_id=0):
    td = datetime.timedelta(days=30)
    api = d2api.APIWrapper(api_key='A0D129625665186D9C1E0BDBF6A0F7A9', parse_response=False)
    dic = fetch_matches(game_type, min_id)  # API key and other details should be handled within fetch_matches
    try:
        with open(f'last_id_new_{game_type}.txt', 'r', encoding='utf-8') as f:
            break_id = int(f.readline())
    except FileNotFoundError:
        break_id = 7500000000
    games = []
    for player_or_match in tqdm(dic):
        if game_type == 'ladder':
            try:
                last_match_time_ig = datetime.datetime.fromisoformat(player_or_match['last_match_time'][:-1])
            except:
                # Handle the KeyError if 'last_match_time' is not present in the JSON
                continue
            if last_match_time - last_match_time_ig < td:
                Steam_id = player_or_match['account_id']
                req = requests.get(f'https://api.opendota.com/api/players/{Steam_id}/matches')
                print(req)
                matches_json = req.json()
                for match in matches_json:
                    print(match)
                    id = int(match['match_id'])
                    max_id = max(max_id, id)
                    if id < break_id:
                        break
                    if match['game_mode'] != 22:
                        continue
                    try:
                        match_details = api.get_match_details(id)
                        match_details = json.loads(match_details)
                        match_details['id'] = id
                        games.append(match_details)
                    except:
                        print('TIMEOUT')
        elif game_type != 'ladder':
            id = int(player_or_match['match_id'])
            min_id = min(min_id, id)
            max_id = max(max_id, id)
            if id < break_id:
                break
            try:
                match_details = api.get_match_details(id)
                match_details = json.loads(match_details)
                match_details['id'] = id
                games.append(match_details)
            except:
                print('TIMEOUT')

    create_draft_data_set(matches=games, game_type=game_type)

    if game_type != 'ladder' and min_id > break_id:
        create_new_dataset('progames', min_id=min_id,max_id=max_id)
        return

    with open(f'last_id_new_{game_type}.txt', 'w', encoding='utf-8') as f:
        f.write(str(max_id))



def role_placement_model(df:pd.DataFrame,wv:KeyedVectors):
    df=df.iloc[:, :10].copy()
    df = df.reset_index(drop=True)
    arr = np.zeros(df.shape)
    gm = GaussianMixture(n_components=5, random_state=69)
    gm.fit(wv.vectors)
    for i,row in df.iterrows():
        r=np.zeros((5))
        d=np.zeros((5))
        prediction = gm.predict_proba(wv[row.values])
        for j,role in enumerate(prediction[:5]):
            for _ in role:
                rl = np.argmax(role)
                if r[rl]==0:
                    r[rl]+=1
                    arr[i][rl] = int(row.values[j])
                    break
                role[rl]=-1
        for j, role in enumerate(prediction[5:]):
            for _ in role:
                rl = np.argmax(np.array(role))
                if d[rl] == 0:
                    d[rl] += 1
                    arr[i][rl+5] = int(row.values[j+5])
                    break
                role[rl]=-1
    return pd.DataFrame(arr)


def create_rivalry_matrix():
    hl = get_Localized_hero_list()
    hero_count = max([d['id'] for d in hl])+1
    arr = np.zeros((hero_count,hero_count))
    for hero_id in hl:
        total_matches = 0
        total_wins = 0
        response = requests.get(f'https://api.opendota.com/api/heroes/{hero_id["id"]}/matchups')
        while response.status_code != 200:
            time.sleep(10)
            print('problem here', response)
            response = requests.get(f'https://api.opendota.com/api/heroes/{hero_id["id"]}/matchups')
        dic = response.json()
        for hero in dic:
            total_wins += float(hero['wins'])
            total_matches += float(hero['games_played'])
            arr[hero_id['id']][hero['hero_id']] = float(hero['wins'])/float(hero['games_played'])
        arr[hero_id['id']][hero_id['id']] = total_wins/total_matches
    pd.DataFrame(arr).to_csv('rivalry_matrix.csv',index=False)


def noise_clear(df:pd.DataFrame):
    all_values = pd.concat([df[col].astype(str) for col in df.columns], ignore_index=True)

    # Get the counts of unique values
    value_counts = all_values.value_counts()

    # Plot the histogram
    plt.figure(figsize=(10, 4))
    value_counts.plot(kind='bar')

    # Set the title and labels
    plt.title('Aggregated Histogram of Unique Values Across All Columns')
    plt.xlabel('Unique Values')
    plt.ylabel('Counts')

    # Show the plot
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()


def remove_rows_with_least_frequent_values(df, least_number):
    # Concatenate all column values into a single series
    only_hero_df = df.iloc[:,:10].copy()
    all_values = pd.concat([df[col].astype(str) for col in only_hero_df.columns], ignore_index=True)

    # Get the counts of unique values
    value_counts = all_values.value_counts()

    # Find the n least frequent values
    least_frequent_values = value_counts.nsmallest(least_number).index.tolist()

    # Create a mask for rows to keep
    mask = df.apply(lambda row: not any(row.astype(str).isin(least_frequent_values)), axis=1)

    # Drop rows that contain any of the n least frequent values
    filtered_df = df[mask]
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df