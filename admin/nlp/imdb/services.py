import csv
import time
from math import log, exp
from os import path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras_preprocessing.sequence import pad_sequences
from collections import defaultdict
from selenium import webdriver
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb

class Imdb(object):
    def __init__(self):
        global train_input, train_target, test_input, test_target, val_input, val_target
        (train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
        train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2,
                                                                            random_state=42)

    def load_data(self):
        print(train_input.shape,test_input.shape)
        print(len(train_input[0]))
        print(len(train_input[1]))
        print(train_input[0])
        print(train_target[:20])

    def draw_histogram(self):
        lengths = np.array([len(x)for x in train_input])
        print(np.mean(lengths), np.median(lengths))
        plt.hist(lengths)
        plt.xlabel('length')
        plt.ylabel('frequency')
        plt.show()

    def training_set(self):
        train_seq = pad_sequences(train_input, maxlen=100)
        print(train_seq.shape)
        print(train_seq[0])
        print(train_input[0][-10:])
        print(train_seq[5])
        val_seq = pad_sequences(val_input, maxlen=100)

    def hook(self):
        self.load_data()
        self.draw_histogram()
        self.training_set()


class NaverMovieService(object):
    def __init__(self):
        global naver_url, savepath, encoding , review_train, k
        savepath = r'C:\Users\AIA\PycharmProjects\djangoProject\admin\nlp\imdb'
        review_train = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\nlp\imdb\review_train.csv"
        encoding = "UTF-8"
        k = 0.5
        self.word_probs = []


    def process(self, new_review):
        service = NaverMovieService()
        service.model_fit()
        result = service.classify(new_review)
        return result

    def crawling(self):
        if path.exists(f"{savepath}\\naver_movie_review.csv") == True:
            data = pd.read_csv(f"{savepath}\\naver_movie_review.csv", header=None)
            data.columns = ['review', 'score']
            result = [print(f"{i+1}. {data['score'][i]}\n{data['review'][i]}\n") for i in range(len(data))]
            return result

        else:
            options = webdriver.ChromeOptions()
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            driver = webdriver.Chrome(r'C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\webcrawler\chromedriver.exe')
            review_data = []

            for page in range(1, 10):
                driver.get(f'https://movie.naver.com/movie/point/af/list.naver?&page={page}')
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                # find_all : 지정한 태그의 내용을 모두 찾아 리스트로 반환
                all_tds = soup.find_all('td', attrs={'class', 'title'})
            # 한 페이지의 리뷰 리스트의 리뷰를 하나씩 보면서 데이터 추출
                for review in all_tds:
                    need_reviews_cnt = 1000
                    sentence = review.find("a", {"class": "report"}).get("onclick").split("', '")[2]
                    if sentence != "":  # 리뷰 내용이 비어있다면 데이터를 사용하지 않음
                        score = review.find("em").get_text()
                        review_data.append([sentence, int(score)])
                        need_reviews_cnt -= 1
            # 현재까지 수집된 리뷰가 목표 수집 리뷰보다 많아진 경우 크롤링 중지
                if need_reviews_cnt < 0:
                    break

            # 다음 페이지를 조회하기 전 1초 시간 차를 두기
            time.sleep(1)

            with open(f'{savepath}\\naver_movie_review.csv', 'w', newline='', encoding=encoding) as f:
                wr = csv.writer(f)
                wr.writerows(review_data)
            driver.close()
            return "크롤링 완료"

    def load_corpus(self):
        corpus = pd.read_table(review_train, sep=",", encoding=encoding)
        corpus = np.array(corpus)
        return corpus

    def count_words(self, train_X):
        counts = defaultdict(lambda: [0, 0])
        for doc, point in train_X:
            if self.isNumber(doc) is False:
                words = doc.split()
                for word in words:
                    counts[word][0 if point > 3.5 else 1] += 1
        return counts

    def isNumber(self, param):
        try:
            float(param)
            return True
        except ValueError:
            return False

    def probability(self, word_probs, doc):
        docwords = doc.split()
        log_prob_if_class0 = log_prob_if_class1 = 0.0
        for word, prob_if_class0, prob_if_class1 in word_probs:
            if word in docwords:
                log_prob_if_class0 += log(prob_if_class0)
                log_prob_if_class1 += log(prob_if_class1)
            else:
                log_prob_if_class0 += log(1.0 - prob_if_class0)
                log_prob_if_class1 += log(1.0 - prob_if_class1)
        prob_if_class0 = exp(log_prob_if_class0)
        prob_if_class1 = exp(log_prob_if_class1)
        return prob_if_class0 / (prob_if_class0 + prob_if_class1)

    def word_probablities(self, counts, n_class0, n_class1, k):
        return [(w,
                 (class0 + k) / (n_class0 + 2 * k),
                 (class1 + k) / (n_class1 + 2 * k))
                for w, (class0, class1) in counts.items()]

    def classify(self, doc):
        return self.probability(word_probs=self.word_probs, doc=doc)

    def model_fit(self):
        train_X = self.load_corpus()
        '''
        '재밌네요': [1,0]
        '별로 재미없어요': [0,1]
        '''
        num_class0 = len([1 for _, point in train_X if point > 3.5])
        num_class1 = len(train_X) - num_class0
        word_counts = self.count_words(train_X)
        # print(f" ************  word_counts is {word_counts}")
        self.word_probs = self.word_probablities(word_counts, num_class0, num_class1, k)

if __name__ == '__main__':
    # ImdbService().hook()
    result = NaverMovieService().process("시간 아깝다. 정말 쓰레기 영화다")
    print(f"긍정률: {result}")
