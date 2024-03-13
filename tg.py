import time

from aiogram import Bot, Dispatcher
import asyncio
import utils
import d2api
import joblib
from scrap import *
import pandas as pd
import xgboost as xgb


clf = xgb.Booster()  # init model
clf.load_model('model.json')

API_KEY = 'A0D129625665186D9C1E0BDBF6A0F7A9'
api = d2api.APIWrapper(api_key=API_KEY, parse_response = True)

API_TOKEN = '6652552891:AAGDAFGbNYcTp0Bb4adRMGR4w8dLecaghbc'
chat_id = '1585954025'


bot = Bot(token=API_TOKEN)
dp = Dispatcher()

async def schedule_messages():
    while True:
        print(time.localtime())
        await send_model_output()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Done~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        await asyncio.sleep(300)


async def send_model_output():
    try:
        vhs = api.get_live_league_games()
    except:
        print('cant get matches')
        raise

    bad_matches = []
    for game in vhs['games']:
        try:
            if (game['scoreboard']['duration']+game['stream_delay_s'])<=(game['stream_delay_s']+400):
                print(game['radiant_team']['team_name'], ' ', game['dire_team']['team_name'], ' ', round(game["scoreboard"]["duration"]/60,2), 'stream delay: ',game['stream_delay_s'])
                continue
            else:
                try:
                    bad_matches.append(game['radiant_team']['team_name'])
                    print(game['radiant_team']['team_name'],' ', game['dire_team']['team_name'] , f' flaged, duration: {game["scoreboard"]["duration"]/60} minutes', 'stream delay: ',game['stream_delay_s'])

                except:
                    rtn = 'err'
                try:
                    bad_matches.append(game['dire_team']['team_name'])
                except:
                    dtn = 'err'
        except:
            #print('Sorry, cant get game duration')
            continue
    matches = get_live_html(bad_matches, Main_url)                          #BB
    if not matches:
        print('no mmatches')
        return
    for game in vhs['games']:
        rc = 0
        dc = 0
        df = pd.DataFrame(columns=['r1',
                                   'r2',
                                   'r3',
                                   'r4',
                                   'r5',
                                   'd1',
                                   'd2',
                                   'd3',
                                   'd4',
                                   'd5', ])
        for player in game['players']:
            if player['side'] != 'radiant' and player['side'] != 'dire':
                continue
            if player['side'] == 'radiant' and int(player['hero']['hero_id'])!=0:
                rc += 1
                id = [int(player['hero']['hero_id'])]
                df[f'r{rc}'] = id
            elif int(player['hero']['hero_id'])!=0:
                dc += 1
                id = [int(player['hero']['hero_id'])]
                df[f'd{dc}'] = id
            else:
                continue
        if df.empty:
            print('df empty')
            rtn = 'err'
            dtn = 'err'
            continue
        else:
            try:
                rtn = game['radiant_team']['team_name']
            except:
                rtn = 'err'
            try:
                dtn = game['dire_team']['team_name']
            except:
                dtn = 'err'
        for match in matches:
            A, B, Game, Coeff_A, Coeff_B = match
            if dtn.strip().capitalize() == A.strip().capitalize() or rtn.strip().capitalize() == B.strip().capitalize():
                out = utils.kelly(100, clf, df, rtn, dtn, coeff=[Coeff_B, Coeff_A])
                await bot.send_message(chat_id, text=(Game + '\n' + str(out)))
                break
            elif dtn.strip().capitalize() == B.strip().capitalize() or rtn.strip().capitalize() == A.strip().capitalize():
                out = utils.kelly(100, clf, df, rtn, dtn,coeff=[Coeff_A,Coeff_B])
                await bot.send_message(chat_id, text=(Game + '\n' + str(out)))
                break
            #else:
            #    rp = utils.get_coef(df,clf)
            #    await bot.send_message(chat_id, text=(rtn +' ' + dtn + '\n' + str(rp) +' '+ str(1-rp)))
            #    break
        await asyncio.sleep(2)

def start_tg():
    asyncio.run(schedule_messages())
#1090 160 for AH