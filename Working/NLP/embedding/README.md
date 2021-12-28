백터화(Vectorization)
=

> 1. Bag of Word
> 2. DTM
> 3. TF-IDF
> 4. LSA & LDA
> 5. soynlp

오늘 알아야할 것은 위의 다섯가지 입니다.

벡터화 즉, 임베딩에대해서 알아보겠습니다.

단어를 벡터화, 혹은 임베딩하는 것은

1. **머신러닝을 활용한 방법**
2. **딥러닝을 활용한 방법**

이 두가지로 나눠서 생각해 볼 수 있습니다.

우선 오늘은 **1. 머신러닝을 활용한 방법**을 알아보겠습니다.

# 1. Bag of Words

**Bag of Words는 특정 문서 내에 특정 단어가 몇번이나 등장하는지를 기준으로 단어벡터를 정의합니다.**

**이러한 방식을 사용하는 임베딩 방법에는** 

**DTM(Document-Term Matrix),** 

**TF-IDF(Term Frequency-Inverse Document Frequency)** 

**가 있습니다.(글씨 순서가 이상하게 계속 바뀌네요;;)**

**보통 언어간의 유사도는 언어벡터간의 코사인 유사도로 계산합니다.**

## **Cosine Similarity**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c4640ecc-5e27-448d-985f-a416d61db3c9/Untitled.png)

**이렇게 계산될 수 있는 임베딩 된 단어벡터를 만들어볼 것입니다.**

# 2. DTM

---

**아래 문장으로 예를 들어볼게요!**

**Doc 1: Intelligent applications creates intelligent business processes**

**Doc 2: Bots are intelligent applications**

**Doc 3: I do business intelligence**

**이렇게 세게의 문서가 있으면 아래와 같이, 문장에 등장한 횟수로 벡터가 만들어지는 거에요.**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/50eab063-0eac-447e-baeb-4a27977d6262/Untitled.png)

**문장간의 코사인 유사도도 구해볼 수 있습니다.**

**아래 문장으로 해볼게요.**

**문서1 : I *like* dog**

**문서2 : I *like* cat**

**문서3 : I *like* cat I *like* cat**

**이를 벡터화 시키면**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f66d5fe3-1be4-47a3-aa68-580999fc19a5/Untitled.png)

**이러한 모습이 되는데, 단어별 벡터도 볼 수 있지만, 문장간의 유사도도 볼 수 있습니다.**

**문서 2와 문서 3은 유사도가 더 높게 나오겠죠?**

```python
import numpy as np
from numpy import dot
from numpy.linalg import norm

doc1 = np.array([0,1,1,1]) # 문서1 벡터
doc2 = np.array([1,0,1,1]) # 문서2 벡터
doc3 = np.array([2,0,2,2]) # 문서3 벡터

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))
```

```python
**print(cos_sim(doc1, doc2)) #문서1과 문서2의 코사인 유사도
print(cos_sim(doc1, doc3)) #문서1과 문서3의 코사인 유사도
print(cos_sim(doc2, doc3)) #문서2과 문서3의 코사인 유사도**
```

**결과가 1인경우는 어느경우일까요?**

구현(from sklearn.feature_extraction.text import CountVectorizer 사용)

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'John likes to watch movies',
    'Mary likes movies too',
    'Mary also likes to watch football games',    
]
vector = CountVectorizer()

print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도수를 기록.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
```

**`[[0 0 0 1 1 0 1 1 0 1]
 [0 0 0 0 1 1 1 0 1 0]
 [1 1 1 0 1 1 0 1 0 1]]
{'john': 3, 'likes': 4, 'to': 7, 'watch': 9, 'movies': 6, 'mary': 5, 'too': 8, 'also': 0, 'football': 1, 'games': 2}`

위 결과를 보고 무슨내용인지 이해가 되면 넘어가세요!**

**하지만 이렇게 벡터화할때, 모든 문장에 자주나오는 단어는 별로 좋지 않을 수 있습니다.(괜찮을 수도 있지만요.) 예를들어 the가 나오지 않는 문서는 그렇게 많지 않을거에요. 이러한 단어는 벡터의 값을 조금 줄여보자는 것이 TF-IDF의 아이디어 입니다.**

# 3. **TF-IDF란**

---

**TF-IDF: DTM에서 각 단어의 중요도를 고려하여 가중치를 줌.**

**TF-IDF는 모든 문서에서 자주 등장하는 단어는 덜 중요하고 판단하고,** 

**특정 문서에서만 자주 등장하는 단어는 더 중요하다고 판단해요.**

**TF-IDF는 Term Frequency-Inverse Document Frequency의 약자입니다. 한국어로 해석하면 '단어 빈도-역문서 빈도' 입니다. 추측하건대, '단어의 빈도' 와 '문서의 빈도의 역수' 를 활용하는 것 같습니다.**

**TF-IDF는 불용어처럼 중요도가 낮으면서 모든 문서에 등장하는 단어들이 노이즈가 되는 것을 완화해 줍니다.** 

### 주의할 점은 TF-IDF를 사용하는 것이 DTM을 사용하는 것보다 성능이 항상 뛰어나지는 않다는 점입니다. TF-IDF를 사용하기 위해서는 우선 DTM을 만든 뒤에 TF-IDF 가중치를 DTM에 적용합니다.

**단어의 빈도를 의미하는 TF는 사실 이미 배웠습니다. DTM이 이미 TF 행렬이기 때문이죠! 그리고 DTM의 각 단어에 IDF 값을 곱해주면 TF-IDF 행렬이 완성됩니다.**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/30183a4b-e528-44e1-a2db-78e5620e8113/Untitled.png)

굳이 log까지 쓴다는 생각이 조금 드는군요..

[https://youtu.be/Rd3OnBPDRbM](https://youtu.be/Rd3OnBPDRbM)

위에거 구하는 방법인데, 저는 안보려구요ㅋㅋ

```python
from math import log
import pandas as pd
print('=3')

docs = [
  'John likes to watch movies and Mary likes movies too',
  'James likes to watch TV',
  'Mary also likes to watch football games',  
]
print('=3')

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()
print('단어장의 크기 :', len(vocab))
print(vocab)

N = len(docs) # 총 문서의 수
N

def tf(t, d):
    return d.count(t)
 
def idf(t):
    df = 0
    for doc in docs:
        df += t in doc    
    return log(N/(df + 1)) + 1
 
def tfidf(t, d):
    return tf(t,d)* idf(t)

result = []
for i in range(N): # 각 문서에 대해서 아래 명령을 수행
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        
        result[-1].append(tf(t, d))
        
tf_ = pd.DataFrame(result, columns = vocab)
tf_

result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index = vocab, columns=["IDF"])
idf_

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        
        result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)
tfidf_
```

직접 구현하면 위와 같지만

sklearn으로 해도 됩니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
  'John likes to watch movies and Mary likes movies too',
  'James likes to watch TV',
  'Mary also likes to watch football games',  
]

tfidfv = TfidfVectorizer().fit(corpus)
vocab = list(tfidfv.vocabulary_.keys()) # 단어장을 리스트로 저장
vocab.sort() # 단어장을 알파벳 순으로 정렬

# TF-IDF 행렬에 단어장을 데이터프레임의 열로 지정하여 데이터프레임 생성
tfidf_ = pd.DataFrame(tfidfv.transform(corpus).toarray(), columns = vocab)
tfidf_
```

# 4. 특이값 분해

[고유값 분해](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-19-%ED%96%89%EB%A0%AC?category=1057680): 이걸 먼저 아셔야 합니다. 사실 몰라도 되지만... 

특이값 분해를 완전히 아는게 아니여서 아는척하기 어렵네요ㅜㅠ

이건 따로 정리좀 해볼게요!!

특이 값 분해: 

<aside>
💡 N×M 크기의 행렬 A를 다음과 같은 3개의 행렬의 곱으로 나타내는 것을 **특이분해(singular-decomposition)** 또는 특잇값 분해(singular value decomposition)라고 한다.

</aside>

$$A=UΣV^T$$

여기에서 U,Σ,V는 다음 조건을 만족해야 한다.

- 대각성분이 양수인 대각행렬이어야 한다. 큰 수부터 작은 수 순서로 배열한다.

$$Σ∈RN×M $$

- UU는 NN차원 정방행렬로 모든 열벡터가 단위벡터이고 서로 직교해야 한다.

$$U∈RN×N$$

- VV는 MM차원 정방행렬로 모든 열벡터가 단위벡터이고 서로 직교해야 한다.

$$V∈RM×M$$

[참조](https://datascienceschool.net/02%20mathematics/03.04%20%ED%8A%B9%EC%9E%87%EA%B0%92%20%EB%B6%84%ED%95%B4.html#id9)

# 5. LSA와 LDA(둘다 잘 안씁니다!)

**LSA: 잠재의미 분석, DTM을 특이값 분해**

**LDA: 문서에 따른 주제(Topic)분포, 주제에 따른 단어분포를 정리.(Topic 개수를 사용자가 정해준다.)**

**LDA를 다루는 코드도 나와는 있는데, 다 만들어진 함수를 이용하는거여서 구체적으로 알기도 어렵고, 출력값도 이상해요ㅋㅋ**

**식은 상당히 복잡하고 잘 쓰지도 않는다고 합니다!**

**그래도 궁금하면 [여기](https://wikidocs.net/30708)를 참고하세요!**

## [LDA데모 써보기](https://lettier.com/projects/lda-topic-modeling/)

# 6. Soynlp

로컬에서 하려면 가상환경에서 

pip install soynlp 

설치해주시면 됩니다!!

글자단위 토크나이저의 하나인데요, 음절단위로 글자를 잘라서 얼마나 연결했을때의 빈도가 가장 많은지 확인해줍니다.

이전에 봤던 wordpiece모델인 bqe랑 똑같은 것 같아요;;

**브랜칭 엔트로피(branching entropy)라는 방식으로 학습을 한다고 합니다.**

### 노드 예시

**제가 어떤 단어를 생각 중인데, 한 문자씩 말해드릴 테니까 매번 다음 문자를 맞추어 보세요.**

**첫 번째 문자는 '*디*'입니다. 다음에 등장할 문자를 맞춰보세요. 솔직히 가늠이 잘 안 가지요? '*디*'로 시작하는 단어가 얼마나 많은데요. 이걸 어떻게 맞추냐구요. 정답은 '*스*' 입니다.**

**이제 '*디스*'까지 나왔네요. '*디스*' 다음 문자는 뭘까요? 벌써 정답 단어를 예측한 분도 있을 테고, 여전히 가늠이 잘 안 가시는 분도 있을 거예요. '*디스카운트*'라는 단어가 있으니까 '*카*'일까? 아니면 '*디스코드*'라는 단어가 있으니까 '*코*'인가? 생각해보니 '*디스코*'가 정답일 수도 있겠네요. 그러면 '*코*'인가? '*디스아너드*'라는 게임이 있으니까 '*아*'?**

**전부 땡땡땡! 이 단어들을 생각하신 분들은 전부 틀렸습니다. 정답은 '*플*'이었습니다.**

**'*디스플*'까지 왔습니다. 다음 문자를 맞춰보세요. 이제 좀 명백해지는군요. 이 정도 되면 헷갈리시는 분들은 거의 없을 거예요. 정답은 '*레*'입니다. '*디스플레*' 다음에는 어떤 문자일까요? 너무 명백해서 문제라고 보기도 어려워졌어요. 정답은 '*이*'입니다. 제가 생각한 단어는 '*디스플레이*'였습니다!**

**저는 지금 브랜칭 엔트로피를 시뮬레이션한 겁니다. 브랜칭 엔트로피를 주어진 문자 시퀀스에서 다음 문자 예측을 위해 헷갈리는 정도라고 비유해 봅시다. 브랜칭 엔트로피의 값은 하나의 완성된 단어에 가까워질수록 문맥으로 인해 정확히 예측할 수 있게 되므로 점차 줄어듭니다. 실습해 볼게요.**

아래 실습파일을 참고하세요!