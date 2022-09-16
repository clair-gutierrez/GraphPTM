import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, InputLayer, Reshape

x_train = np.load('/home/aliak/datasets/k_53_no_labels/x_train_no_labels.npy', allow_pickle=True).astype('float32')
x_validation = np.load('/home/aliak/datasets/k_53_no_labels/x_validation_no_labels.npy', allow_pickle=True).astype('float32')
x_test = np.load('/home/aliak/datasets/k_53_no_labels/x_test_no_labels.npy', allow_pickle=True).astype('float32')

y_train = np.load('/home/aliak/datasets/k_53_no_labels/y_train_no_labels.npy', allow_pickle=True).astype('float32')
y_validation = np.load('/home/aliak/datasets/k_53_no_labels/y_validation_no_labels.npy', allow_pickle=True).astype('float32')
y_test = np.load('/home/aliak/datasets/k_53_no_labels/y_test_no_labels.npy', allow_pickle=True).astype('float32')

x_train = tf.convert_to_tensor(x_train)
x_validation = tf.convert_to_tensor(x_validation)
x_test = tf.convert_to_tensor(x_test)

y_train = tf.convert_to_tensor(y_train)
y_validation = tf.convert_to_tensor(y_validation)
y_test = tf.convert_to_tensor(y_test)

x_train = x_train.reshape(x_train.shape[0], 53, 22, 1)
x_validation = x_validation.reshape(x_validation.shape[0], 53, 22, 1)
x_test = x_test.reshape(x_test.shape[0], 53, 22, 1)

print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(53, 22, 1), activation='relu', kernel_regularizer=l2(1e-06)))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=Adam(1e-04), 
              loss='BinaryCrossentropy', 
              metrics=['accuracy'])
history = model.fit(x=x_train,y=y_train,
          batch_size=50, epochs=400,
          validation_data=(x_validation, y_validation), 
          callbacks=[callback], verbose=1)
model.save('CNN_no_labels.h5')
model.evaluate(x_test, y_test)

from sklearn.metrics import roc_curve, auc, average_precision_score, matthews_corrcoef, f1_score, precision_score

# The predictions are between 0 and 1 because of the sigmoid activation
# function of the last layer in the NN
pred = model.predict(x_test).flatten()

# We can also calculate the AUC to get an estimate how good the NN actually learned
fpr, tpr, thresholds = roc_curve(y_test, pred)
print("AUC:", round(auc(fpr, tpr), 4))
print("AUPRC:", round(average_precision_score(y_test, pred), 4))
pred2 = (pred > 0.5)
pred2 = np.array(pred2)
pred2 = pred2.astype(int)
print("MCC:", round(matthews_corrcoef(y_test, pred2), 4))
print("F1:", round(f1_score(y_test, pred2), 4))
print("Sensitivity:", round(np.average(tpr), 4))
print("Specificity:", round(np.average(1-fpr), 4))
print("Precision:", round(precision_score(y_test, pred2), 4))