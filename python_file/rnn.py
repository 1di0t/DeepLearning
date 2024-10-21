from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import reuters
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

#split data
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

category = np.max(y_train)+1#num of category
print(f"category : {category}")
print(f"Train set : {len(X_train)}")
print(f"Test set : {len(X_test)}")
print(f"X_train example \n:{X_train[0]}\n")#data example
print(f"y_train example \n:{y_train[0]}\n")#label example

#padding - make all data to same length
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test = sequence.pad_sequences(X_test, maxlen=100)

#one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#sturuucture of model
model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(category, activation='softmax'))

#compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

#train
history = model.fit(X_train, y_train, batch_size=20, epochs=200, 
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

#evaluation
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

#draw graph
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='orange', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
