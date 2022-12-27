import numpy as np
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import  keras
from tensorflow.keras.datasets import imdb

class ImdbModel(object):
    def __init__(self):
        global train_input, train_target, test_input, test_target, val_input, val_target,train_seq, val_seq
        (train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
        train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2,
                                                                            random_state=42)
        train_seq = pad_sequences(train_input, maxlen=100)
        val_seq = pad_sequences(val_input, maxlen=100)

    def createmodel(self):
        global model, train_oh, val_oh
        model = keras.Sequential()
        sample_length = 100
        freq_words = 500
        model.add(keras.layers.SimpleRNN(8, input_shape = (sample_length,freq_words)))
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        train_oh = keras.utils.to_categorical(train_seq) #oh is OneHotEncoding
        print(train_oh.shape)
        print(train_oh[0][0][:12])
        print(np.sum(train_oh[0][0]))
        val_oh = keras.utils.to_categorical(val_seq)

    def fit(self):
        rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
        model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', save_best_only = True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience =3, restore_best_weights=True)
        history = model.fit(train_oh, train_target, epochs= 100, batch_size=64,
                            validation_data=(val_oh,val_target),
                            callbacks=[checkpoint_cb,early_stopping_cb])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train','val'])
        plt.show()
        print(train_seq.nbytes, train_oh.nbytes)

    def hook(self):
        self.createmodel()
        self.fit()


class NaverMovieModel(object):
    def __init__(self):
        pass



if __name__ == '__main__':
    ImdbModel().hook()
