import re
from konlpy.tag import Mecab
from collections import Counter
texts = open('../tokenizer/data/naver_movie/ratings.txt','r')
sentence = []
mecab = Mecab() # 빠르고 강력하네요.


counter = 0
for i in texts.readlines():
    i = i.split('\t')[1]
    text = re.sub(r'[^A-zㄱ-ㅎ가-힣ㅏ-ㅣ0-9\s]', '', i)
    sentence.append(mecab.morphs(text))
    counter += 1
    if counter ==500:
        break
what = Counter()
for i in sentence:
    what.update(i)
print(what)
voca = what.most_common(100)

word2idx = {w[0]: index for index,w in enumerate(voca)}

# print(word2idx)

def one_hot_encoding(word, _word2idx):
    onehot_vec = [0]*len(_word2idx)
    idx = _word2idx[word]
    onehot_vec[idx] = 1
    return onehot_vec

print(f'"습니다" : {one_hot_encoding("습니다", word2idx)}')
print(f'"마음" : {one_hot_encoding("마음", word2idx)}')

