import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow import keras


class NumberService(object):
    def __init__(self):
        pass
    def service_model(self,i):
        model = load_model(r"C:/Users/AIA/PycharmProjects/djangoProject/admin/dlearn/save/number_model2.h5")
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        predictions = model.predict(test_images)
        predictions_array, true_label, img = predictions[i], test_labels[i], test_images[i]
        '''
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap=plt.cm.binary)
        '''
        result = np.argmax(predictions_array)
        print(f"예측한 답 : {result}")

menu = ["Exit", "service_model"]  # 1
menu_lambda = {
    "1": lambda x: x.service_model(),
}
if __name__ == '__main__':
    service = NumberService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                menu_lambda[menu](service)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")