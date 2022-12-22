from konlpy.tag import Okt
okt = Okt()
okt.pos('삼성전자 글로벌센터 전자사업부', stem=True)
with open(r'C:\Users\AIA\PycharmProjects\djangoProject\admin\nlp\samsung_report\kr-Report_2018.txt','r',
          encoding='UTF-8') as f:
    texts = f.read()
print(texts)