import os
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
from dataclasses import dataclass
import pandas as pd

@dataclass
class ScrapVO:
    html = ''
    parser = ''
    domain = ''
    query_string = ''
    headers = {}
    tag_name = ''
    fname = ''
    class_names = []
    artists = []
    titles = []
    diction = {}
    df = None

    def dict_to_dataframe(self):
        print(len(self.diction))
        self.df = pd.DataFrame.from_dict(self.diction, orient='index')

    def dataframe_to_csv(self):
        path = r"/admin/dlearn/webcrawler/save/melon_ranking.csv"
        self.df.to_csv(path, sep=',', na_rep="NaN", header=None)




music_menus = ["Exit", #0
                "BugsMusic",#1
                "MelonMusic",#2.
                ]
if __name__=="__main__":
    scrap = ScrapVO()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(music_menus)]
        menu = input('메뉴선택: ')
        if menu == "0":
            print("종료")
            break
        elif menu == "1":
            print("벅스")
            scrap.domain = "https://music.bugs.co.kr/chart/track/day/total?chartdate="
            scrap.query_string = "20221101"
            scrap.parser = "lxml"
            scrap.class_names=["title", "artist"]
            scrap.tag_name = "p"
            BugsMusic(scrap)
        elif menu == "2":
            print("멜론")
            scrap.domain = "https://www.melon.com/chart/index.htm?dayTime="
            scrap.query_string = "2022110909"
            scrap.parser = "lxml"
            scrap.class_names = ["rank01", "rank02"]
            scrap.tag_name = "div"
            MelonMusic(scrap)
        elif menu == "3":
            df = pd.read_csv(f"{static}/save/cop/scp/bugs_ranking.csv")
            print(df)
        else:
            print("해당메뉴 없음")
