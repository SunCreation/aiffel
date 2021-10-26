Shallow Focusing
=
> ## 목차
> ---
> ### 1. 실행 방법   
> ### 2. 목표 및 의의   
> ### 3. 이론 [go](#3-이론)
>   > 1 Semantic Segmentation    
>   > 2 Instance Segmentation   

> ### 4. 코드 분석 [go](#4-코드-분석)
>   > [0 코드 흐름](#0-코드-흐름)   
>   > [1 데이터 준비](#1-데이터-준비-및-전처리)    
>   > [2 모델 설계 및 실험](#2-모델-설계-및-실험)   
>   > [3 모델 평가](#3-모델-평가)   
> ### 5. 아쉬운 점 [go](#5-아쉬운-점)

<br><br><br>

- - - -

## 1. 실행 방법
 1. repository를 내려받는다. ``` git clone https://github.com/SunCreation/aiffel.git ```   
 2. 원하는 사진을 ```aiffel/Working/AI/human_segmentation/images``` 에 위치시킨다.   
 3. ```aiffel/Working/AI/human_segmentation/``` 에서 shallow_focus.py를 실행한다.: ``` python3 shallow_focus.py ```
 4. Working폴더의 위치를 입력한다.  
 5. 다음은 원하는 대로 진행한다.

<br><br><br>

- - - -

## 2. 목표 및 의의

### 1) 완성된 모델을 이용하여, Semantic Segmentation을 구현해본다. 인물과 배경을 분리하여, 인물모드 사진을 만들어보고, 배경 합성들을 해본다.     

### 2) Semantic Segmentation의 한계점을 분석해보고 해결방안을 생각해본다.
<br><br><br><br>

- - - - 
## 3. 이론

<br><br><br><br>

- - - -
## 4. 코드 분석

먼저 다른 모델을 가져와서 그것으로 semantic sagmentation을 진행한다. 
모델은 구글에서 개발한 DeepLab V3+을 사용할 것이다. 


