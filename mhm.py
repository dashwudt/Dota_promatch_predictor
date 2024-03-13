API_TOKEN = '6652552891:AAGDAFGbNYcTp0Bb4adRMGR4w8dLecaghbc'





'''def print_bet_data(file):

    with open(file,'r',encoding='utf-8') as file:
        soup = BeautifulSoup(file,'lxml')
        Team_A        = soup.find_all('div','max-h-[30px] overflow-hidden break-all')[0].text
        Team_B        = soup.find_all('div','max-h-[30px] overflow-hidden break-all')[-1].text
        Match_number  = soup.find_all('div','truncate pb-px text-sm font-normal text-match-detail/scoreboard/label')[-1].text
        try:
            Coefficient_A = soup.find_all('span','text-match-detail/card/header-map'
                                          )[0].find_all_next('div','relative rounded border px-0.5 py-[5px] text-center text-lg font-semibold bg-timeline/'
                                      'card/button-odd/default/normal/background border-timeline/card/button-odd/default/normal/border'
                                      ' text-timeline/card/button-odd/default/normal/label hover:bg-timeline/card/button-odd/default/'
                                      'hover/background hover:border-timeline/card/button-odd/default/hover/border hover:text-timeline'
                                      '/card/button-odd/default/hover/label')[0].text
        except:
            Coefficient_A='666'
        try:
            Coefficient_B = soup.find_all('span','text-match-detail/card/header-map'
                                          )[0].find_all_next('div','relative rounded border px-0.5 py-[5px] text-center text-lg font-semibold bg-timeline/'
                                      'card/button-odd/default/normal/background border-timeline/card/button-odd/default/normal/border'
                                      ' text-timeline/card/button-odd/default/normal/label hover:bg-timeline/card/button-odd/default/'
                                      'hover/background hover:border-timeline/card/button-odd/default/hover/border hover:text-timeline'
                                      '/card/button-odd/default/hover/label')[1].text
        except:
            Coefficient_B='666'
        # printing out current game
        print(Match_number)
        # printing out team names
        print(Team_A,Team_B)
        # printing out coefficients for the current game
        print(Coefficient_A,Coefficient_B)
        return (Team_A,Team_B,Match_number,Coefficient_A,Coefficient_B)

def bet_data(soup):
    Team_A = soup.find_all('div', 'max-h-[30px] overflow-hidden break-all')[0].text
    Team_B = soup.find_all('div', 'max-h-[30px] overflow-hidden break-all')[-1].text
    Match_number = soup.find_all('div', 'truncate pb-px text-sm font-normal text-match-detail/scoreboard/label')[
        -1].text
    try:
        Coefficient_A = soup.find_all('span', 'text-match-detail/card/header-map'
                                      )[0].find_all_next('div',
                                                         'relative rounded border px-0.5 py-[5px] text-center text-lg font-semibold bg-timeline/'
                                                         'card/button-odd/default/normal/background border-timeline/card/button-odd/default/normal/border'
                                                         ' text-timeline/card/button-odd/default/normal/label hover:bg-timeline/card/button-odd/default/'
                                                         'hover/background hover:border-timeline/card/button-odd/default/hover/border hover:text-timeline'
                                                         '/card/button-odd/default/hover/label')[0].text
        Coefficient_B = soup.find_all('span', 'text-match-detail/card/header-map'
                                      )[0].find_all_next('div',
                                                         'relative rounded border px-0.5 py-[5px] text-center text-lg font-semibold bg-timeline/'
                                                         'card/button-odd/default/normal/background border-timeline/card/button-odd/default/normal/border'
                                                         ' text-timeline/card/button-odd/default/normal/label hover:bg-timeline/card/button-odd/default/'
                                                         'hover/background hover:border-timeline/card/button-odd/default/hover/border hover:text-timeline'
                                                         '/card/button-odd/default/hover/label')[1].text
    except:
        Coefficient_A = '666'
        Coefficient_B = '666'
    print(str.strip(Match_number))
    print(str.strip(Team_A), str.strip(Team_B))
    print(str.strip(Coefficient_A), str.strip(Coefficient_B))
    match = (Team_A, Team_B, Match_number, Coefficient_A, Coefficient_B)
    return match
'''