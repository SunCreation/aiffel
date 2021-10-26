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

먼저 다른 모델을 가져와서 그것으로 semantic sagmentation을 진행할 예정입니다.   
모델은 구글에서 개발한 DeepLab V3+을 사용하겠습니다. 

먼저 모델을 만드는데 사용할 클래스를 정의하겠습니다.

```python
class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    # __init__()에서 모델 구조를 직접 구현하는 대신, tar file에서 읽어들인 그래프구조 graph_def를 
    # tf.compat.v1.import_graph_def를 통해 불러들여 활용하게 됩니다. 
    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        with self.graph.as_default():
    	    tf.compat.v1.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    # 이미지를 전처리하여 Tensorflow 입력으로 사용 가능한 shape의 Numpy Array로 변환합니다.
    def preprocess(self, img_orig):
        height, width = img_orig.shape[:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(img_orig, target_size)
        resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img_input = resized_rgb
        return img_input
        
    def run(self, image):
        img_input = self.preprocess(image)

        # Tensorflow V1에서는 model(input) 방식이 아니라 sess.run(feed_dict={input...}) 방식을 활용합니다.
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [img_input]})

        seg_map = batch_seg_map[0]
        return cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR), seg_map

```

좀 울렁증이 생길것 같은 코드입니다.. 
위의 클래스를 이용하여, 완성된 모델을 받아와 사용할 것입니다.


```python
# define model and download & load pretrained weight
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'

model_dir = os.getenv('HOME') + Workingpath + '/Working/AI/human_segmentation/models'
tf.io.gfile.makedirs(model_dir)

print ('temp directory:', model_dir)

download_path = os.path.join(model_dir, 'deeplab_model.tar.gz')
if not os.path.exists(download_path):
    urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
                   download_path)
# 다운로드 완료
MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

img_resized, seg_map = MODEL.run(img_orig)
print ('원본 이미지 크기:', img_orig.shape,'수정 후 이미지 크기:', img_resized.shape,'최대 수치:', seg_map.max())
```


