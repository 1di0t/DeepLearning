from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.models import Sequential


docs = ['맛이 없음','맛이 있음','진짜 맛있어용','천상의 맛','또 시킬래요','맛이 없어요','다신 안 먹어요']#example sentence for sentiment analysis
classes  = array([0,1,1,1,1,0,0])#0:bad, 1:good

#Tokenizing
token = Tokenizer()
token.fit_on_texts(docs)
print(f"{token.word_index}\n")

x = token.texts_to_sequences(docs)
print(f"토큰화 :{x}\n")
print("======================================================================================")

#Padding
padded_x = pad_sequences(x, 3)#fit the length of the sentence to 3
print(f"패딩화 결과 :\n{padded_x}\n")
print("======================================================================================")

word_size = len(token.word_index)+1#Add 1 to add 0 to the front of the array


model = Sequential()
model.add(Embedding(word_size, 8, input_length=3))
model.add(Flatten())#Flatten the input data #3D -> 2D
model.add(Dense(1, activation='sigmoid'))#output layer
model.summary()#학습 전 모델 요약
print("======================================================================================")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_x, classes, epochs=20)

model.summary()#학습 후 모델 요약
print("======================================================================================")

print("\n Accuracy : %.4f"%(model.evaluate(padded_x,classes)[1]))



