import random
import string

import pandas as pd
from sqlalchemy import create_engine

class UserService(object):
    def __init__(self):
        global savepath, filename
# 랜덤 아이디 ,비번(통일) 로 데이터 프레임 만들기
        savepath = r"C:\Users\AIA\PycharmProjects\djangoProject\blog\blog_users\save"
        filename = f"{savepath}\\student_score.sql"

    def create_datas(self):
        f = open(f"{savepath}\\student_score.sql", "w")  # 쓰기모드로 파일 오픈
        datas = []
        for x in range(100):
            one_person_data = self.random_user()
            datas += one_person_data
        # 마지막에 있는 ,(콤마) 를 ; (세미콜론) 으로 변경하기 위한 코드
        datas[-1] = datas[-1][:-1] + ';'
        for data in datas:
            f.write(data)
        f.close()
        engine = create_engine(
            "mysql+pymysql://root:root@localhost:3306/mydb",
            encoding='utf-8')
        datas.to_sql(name='blog_busers',
                  if_exists='append',
                  con=engine,
                  index=False)


    def random_user(self):
        _LENGTH = 8
        string_pool = string.ascii_letters
        result = ""
        for i in range(_LENGTH):
            result += random.choice(string_pool)
        return result




if __name__ == '__main__':
    print(UserService.random_user())
