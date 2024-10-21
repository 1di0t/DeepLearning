from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Embedding(27, 2, input_length=5))#input_length는 입력 시퀀스의 길이
