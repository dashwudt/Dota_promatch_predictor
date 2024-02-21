from selenium import webdriver
from selenium.webdriver import ChromeOptions
import time
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc

Main_url = "https://bifrost.oddin.gg/?referer=https%3A%2F%2Fbetboom.ru%2Fesport&lang=ru&currency=RUB&token=&brandToken=b94ac61d-b060-4892-8242-923bf2303a38"

#/html/body/div[1]/div[3]/div/div[1]/div/div/div[1]/div/div[1]          arrow back
def get_live_html(avoid, url=Main_url):
    matches=[]
    options = ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    #driver = uc.Chrome(headless=False, use_subprocess=True)
    driver.get(url)
    time.sleep(20)
    try:
        live_button = driver.find_element(By.XPATH,'/html/body/div[1]/div[3]/div/div[2]/div/div[2]/div/div[2]')
    except:
        try:
            live_button = driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/div[2]/div/div[2]/div/div[2]')
        except:
            print('Problem actualy here')
            return matches
    live_button.click()
    time.sleep(5)
    is_dota = False
    locator = 2
    while not is_dota:
        try:
                                                        #/html/body/div[1]/div[3]/div[1]/div/div/div/div/div[4]
            dota = driver.find_element(By.XPATH, f'/html/body/div[1]/div[3]/div/div[1]/div/div/div/div/div[{locator}]')
            if dota.accessible_name[2:] == 'Dota 2':
                matches_count = str.strip(dota.accessible_name[:2])
                break
            else:
                locator+=1
        except:
            try:
                dota = driver.find_element(By.XPATH,
                                           f'/html/body/div[1]/div[3]/div[1]/div/div/div/div/div[{locator}]')
                if dota.accessible_name[2:] == 'Dota 2':
                    matches_count = str.strip(dota.accessible_name[:2])
                    break
                else:
                    locator += 1
            except:
                matches_count = '0'
                print('err')
                break

    if int(matches_count):
        dota.click()
        time.sleep(10)
    else:
        print('Problem here!!!')
        return matches
                                       # Счет
    games = []
    if int(matches_count)>1:
        for s in range(1,int(matches_count)+1):
                          #/html/body/div[1]/div[3]/div[2]/div/div[3]/div/div[1]/div[2]/div[1]/div[5]
                          #/html/body/div[1]/div[3]/div[2]/div/div[3]/div/div[2]/div[2]/div[1]/div[5]
                          #/html/body/div[1]/div[3]/div[2]/div/div[3]/div/div[3]/div[2]/div[1]/div[5]
            games.append(f'/html/body/div[1]/div[3]/div[2]/div/div[3]/div/div[{s}]/div[2]/div[1]/div[5]')
    elif int(matches_count) == 1:
                     #/html/body/div[1]/div[3]/div/div[2]/div/div[3]/div/div/div[2]/div[1]/div[5]
        games.append('/html/body/div[1]/div[3]/div[2]/div/div[3]/div/div/div[2]/div[1]/div[5]')
    print(len(games))
    for game in games:
        try:
            driver.find_element(By.XPATH, game).click()
            time.sleep(10)
        except:
            continue
        for _ in range(60):
            try:
                soup = BeautifulSoup(driver.page_source, 'lxml')
                Team_A = soup.find_all('div', 'max-h-[30px] overflow-hidden break-all')[0].text
                #/html/body/div[1]/div[3]/div/div[1]/div/div/div[3]/div[2]/div/div/div[2]/div[1]/div[2]/div
                if soup.find_all('div', 'max-h-[30px] overflow-hidden break-all')[1].text == 'ничья':
                    Team_B = soup.find_all('div', 'max-h-[30px] overflow-hidden break-all')[2].text
                else:
                    Team_B = soup.find_all('div', 'max-h-[30px] overflow-hidden break-all')[1].text

                flag = False
                for bad_name in avoid:
                    if Team_A.strip().capitalize() == bad_name.strip().capitalize() or Team_B.strip().capitalize() == bad_name.strip().capitalize():
                        flag = True
                if flag:
                    print('flagged')
                    break
                Match_number = soup.find_all('div', 'truncate pb-px text-sm font-normal text-match-detail/scoreboard/label')[-1].text
            except:
                time.sleep(1)
                continue
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
                print(str.strip(Match_number))
                print(str.strip(Team_A), str.strip(Team_B))
                print(str.strip(Coefficient_A), str.strip(Coefficient_B))
                matches.append((Team_A, Team_B, Match_number, Coefficient_A, Coefficient_B))
                break
            except:
                time.sleep(1)
                continue
        if int(matches_count)>1:
            try:
                arrow = driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/div[1]/div/div/div[1]/div/div[1]')
                arrow.click()
                time.sleep(5)
            except:
                return matches

    driver.quit()
    return matches

