from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "안녕 클레오파트라 세상에서 제일 가는 포테이토칩 안녕 안녕 클레오파트라 세상에서 제일 가는" 

token = Tokenizer()
token.fit_on_texts([text])
print(f"{token.word_index}\n")#토큰화된 단어와 해당하는 인덱스

x = token.texts_to_sequences([text])#토큰의 인덱스로만 이루어진 리스트
print(text)#원본 텍스트 출력
print(f"{x}\n")#토큰의 인덱스로만 이루어진 리스트 출력

word_size = len(token.word_index)+1#배열 앞에 0을 넣기 위해 +1
x = to_categorical(x, num_classes=word_size)#원핫인코딩
print(f"{x}\n")#원핫인코딩된 리스트 출력