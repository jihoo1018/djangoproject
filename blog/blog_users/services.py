import random
import string

import pandas as pd
from sqlalchemy import create_engine

class UserService(object):
    def __init__(self):
        global savepath, filename, engine
# 랜덤 아이디 ,비번(통일) 로 데이터 프레임 만들기
        savepath = r"C:\Users\AIA\PycharmProjects\djangoProject\blog\blog_users\save"
        filename = f"{savepath}\\student_score.sql"
        engine = create_engine(
            "mysql+pymysql://root:root@localhost:3306/jjyudb",
            encoding='utf-8')


    def insert_users(self):
        dc = self.create_user()
        ls = self.create_users(dc)
        df = self.change_to_df_by_users(ls)
        df.to_sql(name='blog_busers',
                  if_exists='append',
                  con=engine,
                  index=False)

    def create_user(self)->{}:
        string_pool = string.ascii_lowercase
        blog_userid = random.randint(9999,
                                     99999)
        email = str(blog_userid) + "@naver.com"
        nickname = ''.join(random.sample(string_pool, 5))
        password = 0
        return [blog_userid, email, nickname, password]


    def create_users(self)->[]:
        string_pool = string.ascii_lowercase
        blog_userid = random.randint(9999,
                                     99999)
        email = str(blog_userid) + "@naver.com"
        nickname = ''.join(random.sample(string_pool, 5))
        password = 0
        return [blog_userid, email, nickname, password]

    def change_to_df_by_users(self):
        df = [self.create_users() for i in range(100)]
        df = pd.DataFrame(df, columns=['blog_userid', 'email', 'nickname', 'password'])
        df['blog_userid'] = df['blog_userid'].astype(str)
        print(f"df 확인 {df}")
        return df

    def userid_checker(self):
        pass

    def get_users(self):
        pass




if __name__ == '__main__':
    pass
