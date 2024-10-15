from tensorflow.keras.preprocessing.text import Tokenizer

#Tokenization example
#================================================================================================
text = "안녕 클레오파트라 세상에서 제일 가는 포테이토칩 안녕 안녕 클레오파트라 세상에서 제일 가는" 

token = Tokenizer()
token.fit_on_texts([text])# 입력한 텍스트를 단어 빈도수가 높은 순에서 낮은 순으로 순차적으로 정수 인덱스를 부여
print(f"{token.word_index}\n")

sub_text = "세상에서 제일 가는 포테이토칩 안녕"#인코딩을 위한 텍스트


encoded = token.texts_to_sequences([sub_text])[0]#입력한 텍스트를 정수 인덱스로 변환
print(f"[{sub_text}]")#sub_text 원문
print(f"{encoded}\n")#sub_text를 정수 인덱스로 변환한 값

print("============================================================================================\n")
#bag of words example
#================================================================================================
docs = ['햄버거는 맛있는 음식입니다',
        '햄버거는 신선한 재료와 고기가 중요합니다',
        '햄버거는 고기의 익힘 상태에 따라 맛이 달라집니다',]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

print(f"단어 빈도수 : {tokenizer.word_counts}\n")# 단어 빈도수
print(f"문장 수 : {tokenizer.document_count}\n")# 문장 수
print(f"각 단어가 포함된 문장의 수 : {tokenizer.word_docs}\n")# 각 단어가 포함된 문장의 수
print(f"각 단어에 매겨진 인덱스 값 : {tokenizer.word_index}\n")# 각 단어에 매겨진 인덱스 값   

