import keras
from keras import layers
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, classification_report, matthews_corrcoef

EPOCH = 1000
BATCH_SIZE = 64
NUM_LABLES = 13
PATIENCE = 15  # early Stop
save_dir = './2016N/'
path = 'data'
save_model_dir = './save_model/'
x_train = np.load('./data/train_data.npy');
y_train = np.load('./data/train_label.npy');
x_test = np.load('./data/test_data.npy');
y_test = np.load('./data/test_label.npy');
print(x_train.shape)
x_val = x_test[0:6965]
x_test = x_test[6965:]
y_val = y_test[0:6965]
y_test = y_test[6965:]


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


model = keras.Sequential()
model.add(layers.Flatten())  # Exhibit Evaluation Data
model.add(layers.Dense(9, activation='tanh'))
model.add(layers.Dense(12, activation='tanh'))
model.add(layers.Dense(13, activation='softmax'))
model_save_path = save_model_dir + '2016N.h5'

# Checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=model_save_path, save_best_only=True,
                                                save_weights_only=True, monitor='val_accuracy')

# earlystop callback
early_stopping_cb = keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor='val_loss')

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=200, batch_size=128, validation_data=(x_val, y_val),
                    callbacks=[checkpoint_cb, early_stopping_cb])

# draw
font2 = {'size': 10}
plt.figure()
# plt.subplot(2,2,1)
plt.title('2016N-acc')
plt.plot(history.history['accuracy'], label='train accuracy', linewidth=1.5)
plt.plot(history.history['val_accuracy'], label='val accuracy', linewidth=1.5)
plt.legend(prop=font2)
# plt.show()
plt.savefig(save_dir + '2016N-acc.png')

plt.figure()
plt.title(f'2016N-loss')
plt.plot(history.history['loss'], label='train loss', linewidth=1.5)
plt.plot(history.history['val_loss'], label='val loss', linewidth=1.5)
plt.legend(prop=font2)
# plt.show()
plt.savefig(save_dir + '2016N-loss.png')

# load model
model.load_weights(model_save_path)
train_acc, train_loss, val_acc, val_loss = get_best_all(history)
test_loss, test_accuracy = model.evaluate(x_test, y_test)

fp = open(save_dir + f"2016N.txt", 'w+')
fp.write('train_loss: ' + str(train_loss) + '\n' + 'train_acc:  ' + str(train_acc) + '\n')
fp.write('val_loss: ' + str(val_loss) + '\n' + 'val_acc: ' + str(val_acc) + '\n')
fp.write('test_loss: ' + str(test_loss) + '\n' + 'test_accuracy: ' + str(test_accuracy) + '\n')
fp.close()

print('test_loss:', test_loss)
print('test_accuracy', test_accuracy)
# Test set validation
y_test_predict = model.predict(x_test, batch_size=BATCH_SIZE)
y_test_predict = np.array(y_test_predict)
y_test_predict = np.argmax(y_test_predict, axis=1)

report = classification_report(y_test, y_test_predict, digits=5, output_dict=True)
print("MCC=======================================")
print(matthews_corrcoef(y_test, y_test_predict)) 
df = pd.DataFrame(report).transpose()
df.to_csv(save_dir + f"2016N.csv", index=True)
