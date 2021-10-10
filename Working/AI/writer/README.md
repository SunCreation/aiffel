To make AI writer
=
## 목차
### 1. 실행 방법   
### 2. 목표 및 의의   


---
# 2. 목표 및 의의
  - 그럴 듯한 글을 쓰는 인공지능을 만든다.
  - 일단 되는데까지는 해본다. : 인공지능이 만들어주는 글의 길이 조절, 
  - 
이번에는 글을 만드는 인공지능을 만들도록 해보자.



```python
import glob
import os
import re 
import tensorflow as tf
import numpy as np
```

### 먼저 사용할 데이터를 확인해보겠습니다. 
```python

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
print("리스트 예시 5개:\n", txt_name_list[:5], "\n리스트 개수:", len(txt_list))
print("데이터 크기:", len(raw_corpus))
print("Examples:\n", np.array(raw_corpus[:15]))

```
```
데이터 크기: 187088
Examples:
 ['How does a bastard, orphan, son of a whore'
 'And a Scotsman, dropped in the middle of a forgotten spot in the Caribbean by providence impoverished,'
 'In squalor, grow up to be a hero and a scholar? The ten-dollar founding father without a father'
 'Got a lot farther by working a lot harder'
 'By being a lot smarter By being a self-starter'
 'By fourteen, they placed him in charge of a trading charter And every day while slaves were being slaughtered and carted away'
 'Across the waves, he struggled and kept his guard up'
 'Inside, he was longing for something to be a part of'
 'The brother was ready to beg, steal, borrow, or barter Then a hurricane came, and devastation reigned'
 'Our man saw his future drip, dripping down the drain'
 'Put a pencil to his temple, connected it to his brain'
 'And he wrote his first refrain, a testament to his pain Well, the word got around, they said, this kid is insane, man'
 'Took up a collection just to send him to the mainland'
 "Get your education, don't forget from whence you came"
 'And the world is gonna know your name']
```
