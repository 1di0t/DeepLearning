from pre_processing import data_preprocessing
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

#load data
spma_ham  = data_preprocessing.combined_data()

#change data type to numeric
spma_ham['메일종류'] = spma_ham['메일종류'].map({'햄':0, '스팸':1})



#split data
X_train,X_test,y_train,y_test = train_test_split(spma_ham['메일제목'], spma_ham['메일종류'], test_size=0.2, random_state=42)

print(f"X_train : {len(X_train)}")
print(f"X_test : {len(X_test)}")
print(f"y_train : {len(y_train)}")
print(f"y_test : {len(y_test)}")
print(f"X_train example : {X_train[:5]}")
print(f"X_test example : {X_test[:5]}")

#tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
tokenizer.fit_on_texts(X_test)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


print(f"Train set : {len(X_train)}")
print(f"Test set : {len(X_test)}")
print(f"X_train example \n:{X_train[0]}\n")#data example


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
model.add(Dense(2, activation='softmax'))

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
