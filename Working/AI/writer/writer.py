#%%
# import os, re 
# import numpy as np
# import tensorflow as tf  

# # 파일을 읽기모드로 열고
# # 라인 단위로 끊어서 list 형태로 읽어옵니다.
# file_path = os.getenv('HOME') + '/aiffel/Working/AI/writer/data/shakespeare.txt'
# with open(file_path, "r") as f:
#     raw_corpus = f.read().splitlines()

# # 앞에서부터 10라인만 화면에 출력해 볼까요?
# print(raw_corpus[:9])
# # %%
# # 입력된 문장을
# #     1. 소문자로 바꾸고, 양쪽 공백을 지웁니다
# #     2. 특수문자 양쪽에 공백을 넣고
# #     3. 여러개의 공백은 하나의 공백으로 바꿉니다
# #     4. a-zA-Z?.!,¿가 아닌 모든 문자를 하나의 공백으로 바꿉니다
# #     5. 다시 양쪽 공백을 지웁니다
# #     6. 문장 시작에는 <start>, 끝에는 <end>를 추가합니다
# # 이 순서로 처리해주면 문제가 되는 상황을 방지할 수 있겠네요!
# def preprocess_sentence(sentence):
#     sentence = sentence.lower().strip() # 1
#     sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence) # 2
#     sentence = re.sub(r'[" "]+', " ", sentence) # 3
#     sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence) # 4
#     sentence = sentence.strip() # 5
#     sentence = '<start> ' + sentence + ' <end>' # 6
#     return sentence

# # 이 문장이 어떻게 필터링되는지 확인해 보세요.
# print(preprocess_sentence("This @_is ;;;sample        sentence."))
# # %%
# # 여기에 정제된 문장을 모을겁니다
# corpus = []

# for sentence in raw_corpus:
#     # 우리가 원하지 않는 문장은 건너뜁니다
#     if len(sentence) == 0: continue
#     if sentence[-1] == ":": continue
    
#     # 정제를 하고 담아주세요
#     preprocessed_sentence = preprocess_sentence(sentence)
#     corpus.append(preprocessed_sentence)
        
# # 정제된 결과를 10개만 확인해보죠
# corpus[:10]
# # %%
# # 토큰화 할 때 텐서플로우의 Tokenizer와 pad_sequences를 사용합니다
# # 더 잘 알기 위해 아래 문서들을 참고하면 좋습니다
# # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
# # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
# def tokenize(corpus):
#     # 7000단어를 기억할 수 있는 tokenizer를 만들겁니다
#     # 우리는 이미 문장을 정제했으니 filters가 필요없어요
#     # 7000단어에 포함되지 못한 단어는 '<unk>'로 바꿀거에요
#     tokenizer = tf.keras.preprocessing.text.Tokenizer(
#         num_words=7000, 
#         filters=' ',
#         oov_token="<unk>"
#     )
#     # corpus를 이용해 tokenizer 내부의 단어장을 완성합니다.
#     tokenizer.fit_on_texts(corpus)
#     # 준비한 tokenizer를 이용해 corpus를 Tensor로 변환합니다.
#     tensor = tokenizer.texts_to_sequences(corpus)   
#     # 입력 데이터의 시퀀스 길이를 일정하게 맞춰줍니다.
#     # 만약 시퀀스가 짧다면 문장 뒤에 패딩을 붙여 길이를 맞춰줍니다.
#     # 문장 앞에 패딩을 붙여 길이를 맞추고 싶다면 padding='pre'를 사용합니다.
#     tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')  
    
#     print(tensor,tokenizer)
#     return tensor, tokenizer

# tensor, tokenizer = tokenize(corpus)
# # %%
# print(tensor[:3, :10])
# tensor.shape
# # %%
# for idx in tokenizer.index_word:
#     print(idx, ":", tokenizer.index_word[idx])

#     if idx >= 10: break
# # %%
# # tensor에서 마지막 토큰을 잘라내서 소스 문장을 생성합니다
# # 마지막 토큰은 <end>가 아니라 <pad>일 가능성이 높습니다.
# src_input = tensor[:, :-1]  
# # tensor에서 <start>를 잘라내서 타겟 문장을 생성합니다.
# tgt_input = tensor[:, 1:]    

# print(src_input[0])
# print(tgt_input[0])
# # %%
# BUFFER_SIZE = len(src_input)
# BATCH_SIZE = 256
# steps_per_epoch = len(src_input) // BATCH_SIZE

#  # tokenizer가 구축한 단어사전 내 7000개와, 여기 포함되지 않은 0:<pad>를 포함하여 7001개
# VOCAB_SIZE = tokenizer.num_words + 1   

# # 준비한 데이터 소스로부터 데이터셋을 만듭니다
# # 데이터셋에 대해서는 아래 문서를 참고하세요
# # 자세히 알아둘수록 도움이 많이 되는 중요한 문서입니다
# # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# dataset = tf.data.Dataset.from_tensor_slices((src_input, tgt_input))
# dataset = dataset.shuffle(BUFFER_SIZE)
# dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
# dataset
# # %%
# class TextGenerator(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_size, hidden_size):
#         super().__init__()
        
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
#         self.rnn_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
#         self.rnn_2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
#         self.linear = tf.keras.layers.Dense(vocab_size)
        
#     def call(self, x):
#         out = self.embedding(x)
#         out = self.rnn_1(out)
#         out = self.rnn_2(out)
#         out = self.linear(out)
        
#         return out
    
# embedding_size = 256
# hidden_size = 1024
# model = TextGenerator(tokenizer.num_words + 1, embedding_size , hidden_size)
# # %%
# # 데이터셋에서 데이터 한 배치만 불러오는 방법입니다.
# # 지금은 동작 원리에 너무 빠져들지 마세요~
# for src_sample, tgt_sample in dataset.take(1): break

# # 한 배치만 불러온 데이터를 모델에 넣어봅니다
# model(src_sample)
# # %%
# model.summary()
# # %%
# # optimizer와 loss등은 차차 배웁니다
# # 혹시 미리 알고 싶다면 아래 문서를 참고하세요
# # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# # https://www.tensorflow.org/api_docs/python/tf/keras/losses
# # 양이 상당히 많은 편이니 지금 보는 것은 추천하지 않습니다
# optimizer = tf.keras.optimizers.Adam()
# loss = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True,
#     reduction='none'
# )

# model.compile(loss=loss, optimizer=optimizer)
# model.fit(dataset, epochs=30)

# #%%
# def generate_text(model, tokenizer, init_sentence="<start>", max_len=20):
#     # 테스트를 위해서 입력받은 init_sentence도 텐서로 변환합니다
#     test_input = tokenizer.texts_to_sequences([init_sentence])
#     test_tensor = tf.convert_to_tensor(test_input, dtype=tf.int64)
#     end_token = tokenizer.word_index["<end>"]

#     # 단어 하나씩 예측해 문장을 만듭니다
#     #    1. 입력받은 문장의 텐서를 입력합니다
#     #    2. 예측된 값 중 가장 높은 확률인 word index를 뽑아냅니다
#     #    3. 2에서 예측된 word index를 문장 뒤에 붙입니다
#     #    4. 모델이 <end>를 예측했거나, max_len에 도달했다면 문장 생성을 마칩니다
#     while True:
#         # 1
#         predict = model(test_tensor) 
#         # 2
#         predict_word = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)[:, -1] 
#         # 3 
#         test_tensor = tf.concat([test_tensor, tf.expand_dims(predict_word, axis=0)], axis=-1)
#         # 4
#         if predict_word.numpy()[0] == end_token: break
#         if test_tensor.shape[1] >= max_len: break

#     generated = ""
#     # tokenizer를 이용해 word index를 단어로 하나씩 변환합니다 
#     for word_index in test_tensor[0].numpy():
#         generated += tokenizer.index_word[word_index] + " "

#     return generated
# # %%
# generate_text(model, tokenizer, init_sentence="<start> Kill")
# %%

# ----------------------------------------------------


#%%

#%%
import glob
import os
import re
from sklearn.utils import validation 
import tensorflow as tf
import numpy as np

txt_files = os.getenv('HOME') +'/aiffel/Working/AI/writer/data/lyrics/*'
txt_file_path = os.getenv('HOME') +'/aiffel/Working/AI/writer/data/lyrics'
txt_list = glob.glob(txt_files)
txt_name_list = os.listdir(txt_file_path)
raw_corpus = []

# 여러개의 txt 파일을 모두 읽어서 raw_corpus 에 담습니다.
for txt_file in txt_list:
    with open(txt_file, "r") as f:
        raw = f.read().splitlines()
        raw_corpus.extend(raw)
            
print("노래제목 예시 5개:\n", txt_name_list[:5], "\n노래 개수:", len(txt_list))
print("데이터 크기:", len(raw_corpus))
print("Examples:\n", np.array(raw_corpus[:15]))

#%%

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip() # 1
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence) # 2
    sentence = re.sub(r'[" "]+', " ", sentence) # 3
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence) # 4
    sentence = sentence.strip() # 5
    sentence = '<start> ' + sentence + ' <end>' # 6
    
    return sentence
#%%
corpus = []

for sentence in raw_corpus:
    # 우리가 원하지 않는 문장은 건너뜁니다
    if len(sentence) == 0: continue
    if sentence[-1] == ":": continue
    
    # 정제를 하고 담아주세요
    preprocessed_sentence = preprocess_sentence(sentence)
    if len(preprocessed_sentence.split()) > 15 : continue
    corpus.append(preprocessed_sentence)

def tokenize(corpus):
    # 13000단어를 기억할 수 있는 tokenizer를 만들겁니다
    # 우리는 이미 문장을 정제했으니 filters가 필요없어요
    # 13000단어에 포함되지 못한 단어는 '<unk>'로 바꿀거에요
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=13000, 
        filters=' ',
        oov_token="<unk>"
    )
    # corpus를 이용해 tokenizer 내부의 단어장을 완성합니다.
    tokenizer.fit_on_texts(corpus)
    # 준비한 tokenizer를 이용해 corpus를 Tensor로 변환합니다.
    tensor = tokenizer.texts_to_sequences(corpus)   
    # 입력 데이터의 시퀀스 길이를 일정하게 맞춰줍니다.
    # 만약 시퀀스가 짧다면 문장 뒤에 패딩을 붙여 길이를 맞춰줍니다.
    # 문장 앞에 패딩을 붙여 길이를 맞추고 싶다면 padding='pre'를 사용합니다.
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')  
    
    print(tensor,tokenizer)
    return tensor, tokenizer
tensor, tokenizer = tokenize(corpus)
print(tensor[:3, :10])
tensor.shape
#%%
for idx in tokenizer.index_word:
    print(idx, ":", tokenizer.index_word[idx])

    if idx >= 10: break


src_input = tensor[:, :-1]  
# tensor에서 <start>를 잘라내서 타겟 문장을 생성합니다.
tgt_input = tensor[:, 1:]    

print(src_input[0][:10])
print(tgt_input[0][:10])

from sklearn.model_selection import train_test_split as ttst
enc_train, enc_val, dec_train, dec_val = ttst(src_input,
                            tgt_input,
                            test_size=0.2,
                            random_state=21)
#%%
print("Source Train:", enc_train.shape)
print("Target Train:", dec_train.shape)

#%%
BUFFER_SIZE = len(enc_train)
BATCH_SIZE = 256
steps_per_epoch = len(enc_train) // BATCH_SIZE
val_BUFFER_SIZE = len(enc_val)
val_BATCH_SIZE = 256
 # tokenizer가 구축한 단어사전 내 13000개와, 여기 포함되지 않은 0:<pad>를 포함하여 7001개
VOCAB_SIZE = tokenizer.num_words + 1   

dataset = tf.data.Dataset.from_tensor_slices((enc_train, dec_train))
val_dataset = tf.data.Dataset.from_tensor_slices((enc_val, dec_val))

dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset
val_dataset = dataset.shuffle(val_BUFFER_SIZE)
val_dataset = dataset.batch(val_BATCH_SIZE, drop_remainder=True)
val_dataset
#%%

class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.linear = tf.keras.layers.Dense(vocab_size)
        
    def call(self, x):
        out = self.embedding(x)
        out = self.rnn_1(out)
        out = self.rnn_2(out)
        out = self.linear(out)
        
        return out
    
embedding_size = 256
hidden_size = 1024
mywriter = TextGenerator(VOCAB_SIZE, embedding_size , hidden_size)
#%%
# 데이터셋에서 데이터 한 배치만 불러오는 방법입니다.
# 지금은 동작 원리에 너무 빠져들지 마세요~
for src_sample, tgt_sample in dataset.take(1): break

# 한 배치만 불러온 데이터를 모델에 넣어봅니다
mywriter(src_sample)
# #%%
# print(mywriter)
#%%
mywriter.summary()
#%%
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none'
)

mywriter.compile(loss=loss, optimizer=optimizer)
mywriter.fit(dataset, 
        validation_data=val_dataset,
        epochs=10)

#%%
# from sklearn.model_selection import train_test_split as ttst
# enc_train, enc_val, dec_train, dec_val = ttst(tensor,
#                             tokenizer,
#                             test_size=0.2,
#                             random_state=21) ?????


# ---------------------------------------------------


# 입력된 문장을
#     1. 소문자로 바꾸고, 양쪽 공백을 지웁니다
#     2. 특수문자 양쪽에 공백을 넣고
#     3. 여러개의 공백은 하나의 공백으로 바꿉니다
#     4. a-zA-Z?.!,¿가 아닌 모든 문자를 하나의 공백으로 바꿉니다
#     5. 다시 양쪽 공백을 지웁니다
#     6. 문장 시작에는 <start>, 끝에는 <end>를 추가합니다
# 이 순서로 처리해주면 문제가 되는 상황을 방지할 수 있겠네요!
# 이 문장이 어떻게 필터링되는지 확인해 보세요.

# 여기에 정제된 문장을 모을겁니다



# for idx in tokenizer.index_word:
#     print(idx, ":", tokenizer.index_word[idx])

#     if idx >= 10: break

# tensor에서 마지막 토큰을 잘라내서 소스 문장을 생성합니다
# 마지막 토큰은 <end>가 아니라 <pad>일 가능성이 높습니다.
# #%%
# src_input = tensor[:, :-1]  
# # tensor에서 <start>를 잘라내서 타겟 문장을 생성합니다.

# tgt_input = tensor[:, 1:]    

# print(src_input[0])
# print(tgt_input[0])
# #%%
# BUFFER_SIZE = len(src_input)
# BATCH_SIZE = 256
# steps_per_epoch = len(src_input) // BATCH_SIZE

#  # tokenizer가 구축한 단어사전 내 7000개와, 여기 포함되지 않은 0:<pad>를 포함하여 7001개
# VOCAB_SIZE = tokenizer.num_words + 1   

# # 준비한 데이터 소스로부터 데이터셋을 만듭니다
# # 데이터셋에 대해서는 아래 문서를 참고하세요
# # 자세히 알아둘수록 도움이 많이 되는 중요한 문서입니다
# # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# dataset = tf.data.Dataset.from_tensor_slices((src_input, tgt_input))
# dataset = dataset.shuffle(BUFFER_SIZE)
# dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
# dataset
# # %%
# class TextGenerator(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_size, hidden_size):
#         super().__init__()
        
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
#         self.rnn_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
#         self.rnn_2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
#         self.linear = tf.keras.layers.Dense(vocab_size)
        
#     def call(self, x):
#         out = self.embedding(x)
#         out = self.rnn_1(out)
#         out = self.rnn_2(out)
#         out = self.linear(out)
        
#         return out
    
# embedding_size = 256
# hidden_size = 1024
# model = TextGenerator(tokenizer.num_words + 1, embedding_size , hidden_size)
# # %%
# # 데이터셋에서 데이터 한 배치만 불러오는 방법입니다.
# # 지금은 동작 원리에 너무 빠져들지 마세요~
# for src_sample, tgt_sample in dataset.take(1): break

# # 한 배치만 불러온 데이터를 모델에 넣어봅니다
# model(src_sample)
# # %%
# model.summary()
# # %%
# # optimizer와 loss등은 차차 배웁니다
# # 혹시 미리 알고 싶다면 아래 문서를 참고하세요
# # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# # https://www.tensorflow.org/api_docs/python/tf/keras/losses
# # 양이 상당히 많은 편이니 지금 보는 것은 추천하지 않습니다
# optimizer = tf.keras.optimizers.Adam()
# loss = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True,
#     reduction='none'
# )

# model.compile(loss=loss, optimizer=optimizer)
# model.fit(dataset, epochs=30)

#%%
def generate_text(model, tokenizer, init_sentence="<start>", max_len=20):
    # 테스트를 위해서 입력받은 init_sentence도 텐서로 변환합니다
    test_input = tokenizer.texts_to_sequences([init_sentence])
    test_tensor = tf.convert_to_tensor(test_input, dtype=tf.int64)
    end_token = tokenizer.word_index["<end>"]

    # 단어 하나씩 예측해 문장을 만듭니다
    #    1. 입력받은 문장의 텐서를 입력합니다
    #    2. 예측된 값 중 가장 높은 확률인 word index를 뽑아냅니다
    #    3. 2에서 예측된 word index를 문장 뒤에 붙입니다
    #    4. 모델이 <end>를 예측했거나, max_len에 도달했다면 문장 생성을 마칩니다
    while True:
        # 1
        predict = model(test_tensor) 
        # 2
        predict_word = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)[:, -1] 
        # 3 
        test_tensor = tf.concat([test_tensor, tf.expand_dims(predict_word, axis=0)], axis=-1)
        # 4
        if predict_word.numpy()[0] == end_token: break
        if test_tensor.shape[1] >= max_len: break

    generated = ""
    # tokenizer를 이용해 word index를 단어로 하나씩 변환합니다 
    for word_index in test_tensor[0].numpy():
        generated += tokenizer.index_word[word_index] + " "

    return generated
# %%
generate_text(model, tokenizer, init_sentence="<start> man")