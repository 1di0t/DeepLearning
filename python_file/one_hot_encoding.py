from tensorflow.keras.preprocessing.text import Tokenizer 

text = "안녕 클레오파트라 세상에서 제일 가는 포테이토칩 안녕 안녕 클레오파트라 세상에서 제일 가는" 

token = Tokenizer()
token.fit_on_texts([text])# 입력한 텍스트를 단어 빈도수가 높은 순에서 낮은 순으로 순차적으로 정수 인덱스를 부여
print(token.word_index)

sub_text = "세상에서 제일 가는 포테이토칩 안녕"#인코딩을 위한 텍스트


encoded = token.texts_to_sequences([sub_text])[0]#입력한 텍스트를 정수 인덱스로 변환
print(encoded)