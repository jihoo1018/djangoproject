import os

import keras.layers
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


class NumberModel(object):
    def create_model(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print('테스트 정확도:', test_acc)
        loss, accuracy = [], []
        for i in range(10):
            model.fit(x_train, y_train, epochs=1)
            loss.append(model.evaluate(x_test, y_test)[0])
            accuracy.append(model.evaluate(x_test, y_test)[1])
        print(accuracy)
        file_name = os.path.join(os.path.abspath("../save"), "number_model2.h5")
        print(f"저장경로: {file_name}")
        model.save(file_name)


    def test_model(self):
        model = load_model(r"C:/Users/AIA/PycharmProjects/djangoProject/admin/dlearn/save/number_model2.h5")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        plt.imshow(x_test[0].reshape(28, 28))  # 데이터 일자로 펴주기
        plt.show()
        pred = np.argmax(model.predict(x_test),axis=-1)
        print("real:", y_test[0].argmax())  # 7
        print("predict:", pred)



menu = ["Exit", "create_model","test_model"]  # 1
menu_lambda = {
    "1": lambda x: x.create_model(),
    "2": lambda x: x.test_model(),
}
if __name__ == '__main__':
    model = NumberModel()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                menu_lambda[menu](model)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")

