자연어
=

# 0. 왜 자연어일까?
이 공부를 시작하기 전에는 들어보지 못한 말, 자연어(Natural Language)
소설책도, 신문도, 문자나 말로 주고받는 대화도 모두 자연어이다.

그렇다면 자연어가 아닌 언어는 무엇이 있을까.

[자연언어와 인공언어의 처리](https://dukeyang.tistory.com/2)

위 내용을 모두 봐도 좋지만, 주 내용은 

자연어는 문맥의존 문법(Context-sensitive Grammar), 프로그래밍 언어는 문맥자유 문법(Context-free Grammar)을 따른다는 내용이다.

자연어는 문맥에 의존해야만 올바로 해석할 수 있는 경우들이 존재한다.
(사실 프로그래밍 언어도 마찬가지 아닌가?)

# 0. 전처리
언제나 0번은 전처리...

 - 자연어의 노이즈
 자연어에는 노이즈가 잔뜩 있습니다.
 1. 불완전한 문장으로 구성된 대화의 경우
 예1) A: "아니아니" "오늘이 아니고" "내일이지" / B: "그럼 오늘 사야겠네. 내일 필요하니까?"
 2. 문장의 길이가 너무 길거나 짧은 경우
 예1) A: "ㅋㅋㅋ", "ㅠㅠㅠ"
 예2) A: "이 편지는 영국에서부터 시작되어…"
 3. 채팅 데이터에서 문장 시간 간격이 너무 긴 경우
 예1) A: "겨울왕국2" / B: "보러가자" / A: "엊그제 봤는데 잼씀" / B: "오늘 저ㄴ 아니 ㅡㅡ"
 예2) A: "나 10만원만 빌려줄 수 있어?" / …… / B: "아 미안 지금 봤다 아직 필요해?"
 4. 바람직하지 않은 문장의 사용
 욕설, 오타


# 1. Tokenizer

## $\bullet$ Khaiii 설치해서 사용해보기


윈도우 유저는 wsl과 우분투를 설치 후 진행할 수 있습니다.
먼저 아래와 같이 작업디렉토리를 구성해 봅시다.

$ mkdir -p ~/aiffel/text_preprocess
아래 스크립트를 따라서 카카오에서 제공하는 Khaiii 형태소 분석기 설치를 진행합니다. Khaiii의 설치 과정에서 컴파일 과정을 포함하여 시간이 다소 소요됩니다. 설치가 완료되면 테스트 문장 입력을 요구하는데 정상적으로 설치가 되었는지 확인차 몇 가지 문장을 입력해 보시기 바랍니다.

$ sudo apt install cmake   # Khaiii의 빌드 과정을 위해 cmake를 필요로 합니다.
$ pip install torch     # Khaiii는 구동을 위해 파이토치를 필요로 합니다. 
$ cd ~/aiffel/text_preprocess
$ git clone https://github.com/modulabs/khaiii_wrapper.git
$ cd ~/aiffel/text_preprocess/khaiii_wrapper/khaiii_pos
$ ./install_khaiii.sh

이걸로 안될시(위에거는 아이펠 클라우드에서 하면 한방!)
아래를 보고 설치하세요
[here](https://yj-79.tistory.com/11)
[here](https://somjang.tistory.com/entry/Ubuntu-CMake-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95)
sodu apt-get install ssh libssl-dev
cmake 쓸때 우분투 20.04버전에서는 아래처럼 써야 가능합니다...
CXXFLAGS="-Wno-error=deprecated-copy" cmake ..
이거보고 설치
참고로 우분투를 이제 막 설치했으면...[이거 참고해도 좋습니다.](https://wnsgml972.github.io/setting/2019/05/07/wsl/)
이걸 설치하고 진행합니다.
쓴 명령어들

sudo apt-get install cmake (이거로 안되면... 위쪽에 cmake 설치 링크부터.. 하지만 기본적으로 원래 있는 거 같아요.)
cmake --version
 - khaiii 형태소 분석기 설치파일 가져오기
git clone https://github.com/kakao/khaiii.git 
cd khaiii
mkdir build
cd build
sudo cmake .. 를 20.04버전에서는 CXXFLAGS="-Wno-error=deprecated-copy" cmake .. 로
sudo make all
sudo make resource
sudo make install
sudo make package_python
cd package_python
pip install .

## $\bullet$ MeCab 설치해서 써보기..

$ pip install konlpy
$ cd ~/aiffel/text_preprocess
$ git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
$ cd Mecab-ko-for-Google-Colab
$ sudo bash install_mecab-ko_on_colab190912.sh(이거는 가상환경이어도.. 아니여도.. sudo를 쓰자)

필요하면 가상환경 만들어서 하는게? 좋을 것 같아요.



위에꺼보다는 설치하기 매우 좋네요^^ → 취소
Mecab은 마음에 안들면 형태소 사전 추가 가능

아래쪽 #3에서 함 써봐요.

# 2. Embedding
이번에는 별 내용 없습니다..ㅎㅎ 오늘의 주제는 토큰화이기 때문에.
만약 100개의 단어를 256차원의 속성으로 표현하고 싶다면

embedding_layer = tf.keras.layers.Embedding(input_dim=100, output_dim=256)

이런식으로 쓰면 된다.

두 고차원 벡터의 유사도는 코사인 유사도(Cosine Similarity) 를 통해 구할 수 있다.

[이건 봐야할 것 같아요](https://wikidocs.net/22660)
임베딩 벡터 학습 방식입니다.
CBOW(Continuous Bag of Words)와 Skip-Gram 방식이 있는데, Skip-Gram이 더 좋대요!

CBOW: 주변단어들로부터 해당단어를 추론하는 방식으로 학습

Skip-Gram: 해당 단어로 주변단어들을 추론하면서 학습


# 3. 다양한 Tokenizer 사용해보기
일단 [이걸](https://konlpy-ko.readthedocs.io/ko/v0.4.3/)읽어야해요.
그다음엔 [이거](https://iostream.tistory.com/144)


위에 있는거 쭉 설치했으면 아래 코드들로 확인해볼 수 있습니다.

```python
from konlpy.tag import Hannanum,Kkma,Komoran,Mecab,Okt


import khaiii

api = khaiii.KhaiiiApi()
api.open()

class Khaiii():
    def pos(self, phrase, flatten=True, join=False):
        """POS tagger.

        :param flatten: If False, preserves eojeols.
        :param join: If True, returns joined sets of morph and tag.

        """
        sentences = phrase.split('\n')
        morphemes = []
        if not sentences:
            return morphemes

        for sentence in sentences:
            for word in api.analyze(sentence):
                result = [(m.lex, m.tag) for m in word.morphs]
                if join:
                    result = ['{}/{}'.format(m.lex, m.tag) for m in word.morphs]

                morphemes.append(result)

        if flatten:
            return sum(morphemes, [])

        return morphemes


tokenizer_list = [Hannanum(),Kkma(),Komoran(),Mecab(),Okt(),Khaiii()]

kor_text = '코로나바이러스는 2019년 12월 중국 우한에서 처음 발생한 뒤 전 세계로 확산된, 새로운 유형의 호흡기 감염 질환입니다. 등등 넣고싶은 문장은 너봅시다. 오타 있어도 되영'

for tokenizer in tokenizer_list:
    print('[{}] \n{}'.format(tokenizer.__class__.__name__, tokenizer.pos(kor_text)))


```



추가로 



씹어먹어야한다!


엑소 브레인?