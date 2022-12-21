import csv
import os.path
import urllib
from urllib.request import urlopen

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

from admin.dlearn.webcrawler.models import ScrapVO


class ScrapService(object):
    def __init__(self):
        global driverpath, naver_url, savepath, encoder, file
        driverpath = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\webcrawler\chromedriver"
        savepath = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\webcrawler\save"
        naver_url = "https://movie.naver.com/movie/sdb/rank/rmovie.naver"
        encoder = "UTF-8"
        file = f"{savepath}\\naver_movie_review.csv"

    def bugs_music(self, arg): #beautifulsoup 기본 크롤링
        soup = BeautifulSoup(urlopen(arg.domain + arg.query_string), 'lxml')
        title = {"class": arg.class_names[0]}
        artist = {"class": arg.class_names[1]}
        titles = soup.find_all(name=arg.tag_name, attrs=title)
        titles = [i.find('a').text for i in titles]
        artists = soup.find_all(name=arg.tag_name, attrs=artist)
        artists = [i.find('a').text for i in artists]
        [print(f"{i}위 {j} : {k}")  # 디버깅
         for i, j, k in zip(range(1, len(titles)), titles, artists)]
        diction = {}  # dict 로 변환
        for i, j in enumerate(titles):
            diction[j] = artists[i]
        arg.diction = diction
        arg.dict_to_dataframe()
        arg.dataframe_to_csv()  # csv파일로 저장

    def melon_music(self, arg): #beautifulsoup 기본 크롤링
        soup = BeautifulSoup(
            urlopen(urllib.request.Request(arg.domain + arg.query_string, headers={'User-Agent': "Mozilla/5.0"})),
            "lxml")
        title = {"class": arg.class_names[0]}
        artist = {"class": arg.class_names[1]}
        titles = soup.find_all(name=arg.tag_name, attrs=title)
        titles = [i.find('a').text for i in titles]
        artists = soup.find_all(name=arg.tag_name, attrs=artist)
        artists = [i.find('a').text for i in artists]
        [print(f"{i}위 {j} : {k}")  # 디버깅
         for i, j, k in zip(range(1, len(titles)), titles, artists)]
        diction = {}  # dict 로 변환
        for i, j in enumerate(titles):
            diction[j] = artists[i]
        arg.diction = diction
        arg.dict_to_dataframe()
        arg.dataframe_to_csv()  # csv파일로 저장

    def naver_movie_review(self):
        if os.path.exists(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\webcrawler\save\naver_movie_rank.csv") == True:
            dt = pd.read_csv(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\webcrawler\save\naver_movie_rank.csv")
            # result = [f'{i + 1}위 : {j} ,' for i, j in enumerate(dt)]
            result = []
            for i, j in enumerate(dt):
                i = {"rank" : i+1, "title" : j}
                result.append(i)
                print(result)
            return result

        else:
            options = webdriver.ChromeOptions()
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            driver = webdriver.Chrome(driverpath)

            # Bluetooth: bluetooth_adapter_winrt.cc:1074 Getting Default Adapter failed error
            # https://darksharavim.tistory.com/606 → 해결
            # csv 있으면 arlet, 없으면 크롤링

            driver.get(naver_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            all_divs = soup.find_all('div', attrs={'class', 'tit3'})
            products = [[div.a.string for div in all_divs]]
            with open(f'{savepath}\\naver_movie_rank.csv', 'w', newline='', encoding=encoder) as f:
                wr = csv.writer(f)
                wr.writerows(products)
            driver.close()
            return "크롤링 완료"
