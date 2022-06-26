import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, recall_score,matthews_corrcoef
from collections import Counter
import os
import ipykernel  # If you use pycharm


def get_best_all(history):
    best_val_acc_list = history.history['val_accuracy']
    best_val_acc = max(best_val_acc_list)
    best_val_acc_index = best_val_acc_list.index(best_val_acc)

    epoch_index = best_val_acc_index + 1
    print('epoch_index: ', epoch_index)

    best_train_acc = history.history['accuracy'][best_val_acc_index]
    best_train_loss = history.history['loss'][best_val_acc_index]
    best_val_acc_test = history.history['val_accuracy'][best_val_acc_index]
    best_val_loss = history.history['val_loss'][best_val_acc_index]

    return best_train_acc, best_train_loss, best_val_acc, best_val_loss


def model_Sequential(x_train, batch_size, num_lables):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(x_train.shape[1], 1, x_train.shape[3]), batch_size=batch_size),
        keras.layers.Conv2D(8, kernel_size=(5, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format='channels_last'),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
        keras.layers.Conv2D(16, kernel_size=(5, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format="channels_last"),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
        keras.layers.Conv2D(8, kernel_size=(5, 1), activation='relu', strides=(1, 1), padding='same',
                            data_format="channels_last"),
        # keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None),
        keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
        # keras.layers.Dropout(0.1),
        keras.layers.Flatten(),
        keras.layers.Dense(num_lables, activation='softmax')
    ])
    print(model.summary())
    return model


EPOCH = 1000
BATCH_SIZE = 32
NUM_LABLES = 13
PATIENCE = 15  # early Stop
save_dir = './2018CNN2/'
path = 'data'
save_model_dir = './save_model/'

if __name__ == '__main__':
    x_train = np.load('./data/train_data.npy');
    y_train = np.load('./data/train_label.npy');
    X_test = np.load('./data/test_data.npy');
    Y_test = np.load('./data/test_label.npy');

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(X_test, axis=2)
    x_val = x_test[0:6965]
    x_test = x_test[6965:]
    # reshape
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = Y_test.reshape((Y_test.shape[0], 1))
    y_val = y_test[0:6965]
    y_test = y_test[6965:]

    y_test_true = y_test
    y_train = keras.utils.to_categorical(y_train, NUM_LABLES)
    y_test = keras.utils.to_categorical(y_test, NUM_LABLES)
    y_val = keras.utils.to_categorical(y_val, NUM_LABLES)

    # cut
    train_num = x_train.shape[0] % BATCH_SIZE
    val_num = x_val.shape[0] % BATCH_SIZE
    test_num = x_test.shape[0] % BATCH_SIZE
    if train_num != 0:
        x_train = x_train[0:-train_num]
        y_train = y_train[0:-train_num]

    if val_num != 0:
        x_val = x_val[0:-val_num]
        y_val = y_val[0:-val_num]

    if test_num != 0:
        x_test = x_test[0:-test_num]
        y_test = y_test[0:-test_num]
        y_test_true = y_test_true[0:-test_num]

    print('x_train: ', x_train.shape)
    print('x_test: ', x_test.shape)
    print('y_train: ', y_train.shape)
    print('y_test: ', y_test.shape)
    print('x_val: ', x_val.shape)
    print('y_val: ', y_val.shape)

    cnn_model = model_Sequential(x_train=x_train, batch_size=BATCH_SIZE, num_lables=NUM_LABLES)

    model_save_path = save_model_dir + '2018CNN2.h5'

    # Checkpoint callback
    checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=model_save_path, save_best_only=True,
                                                    save_weights_only=True, monitor='val_accuracy')

    # earlystop callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor='val_loss')

    # compile 
    cnn_model.compile(optimizer=keras.optimizers.Adam(clipnorm=1., lr=0.0001),
                      loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # train
    history = cnn_model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1,
                            validation_data=(x_val, y_val), callbacks=[checkpoint_cb, early_stopping_cb])

    early_stopping_epoch = early_stopping_cb.stopped_epoch - PATIENCE + 1
    print('Early stopping epoch: ' + str(early_stopping_epoch))
    # draw
    font2 = {'size': 10}
    plt.figure()
    # plt.subplot(2,2,1)
    plt.title('2018CNN2-acc')
    plt.plot(history.history['accuracy'], label='train accuracy', linewidth=1.5)
    plt.plot(history.history['val_accuracy'], label='val accuracy', linewidth=1.5)
    plt.legend(prop=font2)
    # plt.show()
    plt.savefig(save_dir + '2018CNN2-acc.png')

    plt.figure()
    plt.title(f'2018CNN2-loss')
    plt.plot(history.history['loss'], label='train loss', linewidth=1.5)
    plt.plot(history.history['val_loss'], label='val loss', linewidth=1.5)
    plt.legend(prop=font2)
    # plt.show()
    plt.savefig(save_dir + '2018CNN2-loss.png')

    # load model
    cnn_model.load_weights(model_save_path)
    train_acc, train_loss, val_acc, val_loss = get_best_all(history)
    test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test)

    fp = open(save_dir + f"2018CNN2.txt", 'w+')
    fp.write('train_loss: ' + str(train_loss) + '\n' + 'train_acc:  ' + str(train_acc) + '\n')
    fp.write('val_loss: ' + str(val_loss) + '\n' + 'val_acc: ' + str(val_acc) + '\n')
    fp.write('test_loss: ' + str(test_loss) + '\n' + 'test_accuracy: ' + str(test_accuracy) + '\n')
    fp.close()

    print('test_loss:', test_loss)
    print('test_accuracy', test_accuracy)
    # Test set validation
    y_test_predict = cnn_model.predict(x_test, batch_size=BATCH_SIZE)
    y_test_predict = np.array(y_test_predict)
    y_test_predict = np.argmax(y_test_predict, axis=1)

    report = classification_report(y_test_true, y_test_predict, digits=5, output_dict=True)
    print("MCC=======================================")
    print(matthews_corrcoef(y_test_true, y_test_predict))
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_dir + f"2018CNN2.csv", index=True)
