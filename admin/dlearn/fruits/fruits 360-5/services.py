import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

class FruitsService():
    def __init__(self):
        global class_names, trainpath,testpath,Apple_Braeburn_Test, \
            Apple_Crimson_Snow_Test ,Apple_Golden_1_Test,Apple_Golden_2_Test, \
            Apple_Golden_3_Test,Apple_Braeburn_Train,Apple_Crimson_Snow_Train, \
            Apple_Golden_1_Train,Apple_Golden_2_Train,Apple_Golden_3_Train, savepath
        savepath = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\fruits\fruits 360-5\save"
        testpath = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\fruits\fruits 360-5\Test"
        trainpath = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\fruits\fruits 360-5\Training"
        Apple_Braeburn_Test = f"{testpath}\\Apple Braeburn"
        Apple_Crimson_Snow_Test = f"{testpath}\\Apple Crimson Snow"
        Apple_Golden_1_Test = f"{testpath}\\Apple Golden 1"
        Apple_Golden_2_Test = f"{testpath}\\Apple Golden 2"
        Apple_Golden_3_Test = f"{testpath}\\Apple Golden 3"
        Apple_Braeburn_Train = f"{trainpath}\\Apple Braeburn"
        Apple_Crimson_Snow_Train = f"{trainpath}\\Apple Crimson Snow"
        Apple_Golden_1_Train = f"{trainpath}\\Apple Golden 1"
        Apple_Golden_2_Train = f"{trainpath}\\Apple Golden 2"
        Apple_Golden_3_Train = f"{trainpath}\\Apple Golden 3"



    def hook(self):
        self.show_apple()
        self.apple_braeburn()

    def show_apple(self):
        img = tf.keras.preprocessing.image.load_img \
            (f'{trainpath}\\Apple Golden 3/0_100.jpg')
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    def apple_braeburn(self):
        batch_size = 32
        img_height = 100
        img_width = 100
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            trainpath,
            validation_split=0.3,
            subset="training",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            trainpath,
            validation_split=0.3,
            subset="validation",
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        class_names = train_ds.class_names
        print(class_names)
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            testpath,
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        test_ds1 = tf.keras.preprocessing.image_dataset_from_directory(
            testpath,
            seed=1,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False)
        type(test_ds)
        y = np.concatenate([y for x, y in test_ds], axis=0)
        print(y)
        y = np.concatenate([y for x, y in test_ds1], axis=0)
        print(y)
        x = np.concatenate([x for x, y in test_ds1], axis=0)
        print(x[0])
        plt.figure(figsize=(3, 3))
        plt.imshow(x[0].astype("uint8"))
        plt.title(class_names[y[0]])
        plt.axis("off")
        plt.show()


        BUFFER_SIZE = 10000
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        train_ds = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        type(train_ds)
        num_classes = 5
        model = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(.50),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Dropout(.50),
            layers.Flatten(),
            layers.Dense(500, activation='relu'),
            layers.Dropout(.50),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.summary()
        model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
        checkpointer = ModelCheckpoint(f'{savepath}\\CNNClassifier.h5', save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, monitor='val_accuracy',
                                                          restore_best_weights=True)
        epochs = 20

        history = model.fit(
            train_ds,
            batch_size=batch_size,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[checkpointer, early_stopping_cb]
        )
        len(history.history['val_accuracy'])
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, 9 + 1)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()





if __name__ == '__main__':
    FruitsService().hook()