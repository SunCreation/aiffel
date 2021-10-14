```Searching a celebrity looks like me!```
=
본 문서는 공사중입니다.
> ## 목차
> ---
> ### 1. 실행 방법   
> ### 2. 목표 및 의의   
> ### 3. 이론 [go](#3-이론)
>   > 1 embedding이란?
>   > 2 face_recognition 라이브러리
>   > 3 triplet loss
> ### 4. 코드 분석 [go](#4-코드-분석)
>   > [0 코드 흐름](#0-코드-흐름)   
>   > [1 데이터 준비](#1-데이터-준비-및-정제)    
>   > [2 모델 설계](#2-모델-설계)   
>   > [3 학습!](#3-학습)   
>   > [4 모델 평가](#4-모델-평가)   
> ### 5. 아쉬운 점 [go](#5-아쉬운-점)


# 1. 실행 방법
 1. 필요 라이브러리: matplotlib, face_recognition, numpy
 2. 

# 2. 목표 및 의의
 1. 나와 닮은 사진 연예인을 찾을 수 있다.
 2. 원하는 사진과 닮은 연예인 사진을 찾는 프로그램을 만든다.(조건: 프로그램이 연예인 사진을 모두 들고 있으면 곤란하지 않을까..?)
 3. embedding에 대해서 이해한다.
 4. triplet loss손실함수에 대해 이해해본다.
# 3. 이론
## 1. embedding이란?
embedding은 원래 있던 공간과 다른 위상을 같는 공간으로 대상을 변형시키는 변환을 의미한다. 먼저 수학적으로 갖는 의미가 이러한데, 특히 머신러닝에서 많이 사용되는 용어인 듯 해요. 개인적으로 이해하기로는 특정 대상이 같는 정사영, projection과 그 의미가 일맥상통하는 부분이 있다고 생각했어요. 먼저 원래 대상과 다른 차원에서 대상을 본다는 점과 그 대상의 원하는 특성을 뽑아내서 본다는 부분이 그러합니다. 사실 저번 프로젝트에서도 
![이미지](https://d3s0tskafalll9.cloudfront.net/media/original_images/E-11-3.jpg)

https://d3s0tskafalll9.cloudfront.net/media/original_images/E-11-4.png

https://d3s0tskafalll9.cloudfront.net/media/images/E-11-5.max-800x600.png

# 4. 코드 분석
[목차](#searching-a-celebrity-looks-like-me)
## 0. 코드 흐름
아래 코드는 searching.py의 코드 설명으로, 위 실행파일과 다릅니다.(사유: 사진용량 답이 없습니다.)
    
이번 프로젝트에서는 위에서 설명한 embedding 과 triplet loss를 이용하여 서로 닮은 사진을 가려내는 재미있는 일을 해보겠습니다. 
[목차](#searching-a-celebrity-looks-like-me)
## 1. 데이터 준비
#### 이번 프로젝트에서 준비할 자료는 많은 연예인 사진과, 또 그 연예인 사진과 비교할 사진입니다. 또 연예인 사진은 사진 제목이 연예인 이름으로 라벨링이 잘 되어있어야, 잘 사용할 수 있습니다. 
####
```python
import os

dir_path = os.getenv('HOME')+'/Working/AI/face_embedding/images'
test_dir_path = os.getenv('HOME')+'/Working/AI/face_embedding/test'
file_list = os.listdir(dir_path)
test_list = os.listdir(test_dir_path)

print ("file_list: {}".format(file_list))
```



```python
import matplotlib.pyplot as plt
import matplotlib.image as img

#Set figsize here
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,10))

# flatten axes for easy iterating
for i, ax in enumerate(axes.flatten()):
    image = img.imread(dir_path+'/'+file_list[i])
    ax.imshow(image)
plt.show()
```


[목차](#searching-a-celebrity-looks-like-me)

```
닮은 꼴 연예인 순위 발표!!
1위 : 박원상, - 얼굴 거리:  0.40489069421258506
2위 : 박지일, - 얼굴 거리:  0.4157318085194855
3위 : 이종혁, - 얼굴 거리:  0.43105342754771403
4위 : 백도빈, - 얼굴 거리:  0.4388600586663692
5위 : 하정우, - 얼굴 거리:  0.4460418737626919
6위 : 윤주상, - 얼굴 거리:  0.4462334120248691
7위 : 윤세웅, - 얼굴 거리:  0.455455605967715
8위 : 최종윤, - 얼굴 거리:  0.460045631672175
9위 : 박철민, - 얼굴 거리:  0.46207023889806076
10위 : 김민기, - 얼굴 거리:  0.4639429521615208
11위 : 임한별, - 얼굴 거리:  0.4695243868889826
12위 : 양택조, - 얼굴 거리:  0.4704170293181993
13위 : 차인표, - 얼굴 거리:  0.47098473156672643
14위 : 백년설, - 얼굴 거리:  0.47291298489487166
15위 : 이윤상, - 얼굴 거리:  0.47323621485487805
16위 : 전무송, - 얼굴 거리:  0.47403851201881625
17위 : 이승철, - 얼굴 거리:  0.4741976547517154
18위 : 해창, - 얼굴 거리:  0.47482230657532487
박원상을(를) 가장 닮았네요:)
```


# 5. 아쉬운 점