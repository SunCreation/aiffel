#%%
# 
import warnings
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import cv2
import numpy as np
import os
import tarfile
import urllib
import re

from matplotlib import pyplot as plt
import tensorflow as tf

import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('frot', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    pass # print('Unknown system... sorry~~~~',platform.system())
# [f.fname for f in font_manager.fontManager.ttflist if 'Nanum' in f.name]

plt.rcParams["font.family"] = 'NanumSquare'

# %%

Workingpath = input("\n\n\n\n\n사용자의 home 디렉토리에서 Working파일 사이의 경로를 입력하세요.\
    \n(ex: /aiffel/assignment   ,  Working파일이 Home 디렉토리에 있다면, 그냥 Enter를 누르세요.) :")
image_path = os.getenv('HOME')+Workingpath+'/Working/AI/human_segmentation/images/'
image_list = os.listdir(image_path)

def select_img(image_list):
    n=0
    while n==0 :
        print("어떤 사진을 선택할까요?\n")
        num=1

        for i in image_list:
            print(str(num)+'번:', i)
            num += 1    

        try:
            a = input("사진에 해당하는 번호를 입력하세요:")
            b = int(a[:re.search('[0-9]+',a).end()])

        except:
            print('\n\n\n\n숫자를 골라주세요!\n\n\n\n\n')
            continue
        n=1

    return b

n=0
while n==0:
    print('Shallow focusing 할 사진을 골라봅시다.')
    num = select_img(image_list)
    image_name = image_list[num-1]
    img_orig = cv2.imread(image_path + image_name)
    img_show = img_orig.copy()
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)    
    plt.imshow(img_rgb)
    only_imagename = os.path.splitext(image_name)[0]
    plt.title(only_imagename)
    plt.show()
    end=input("이 사진으로 하시겠어요? (y/n)\n\n:")
    if end == 'y' or end == 'yes' or end == 'Y':
        n=1


# img_path = os.getenv('HOME')+Workingpath+'/Working/AI/human_segmentation/images/인물사진.JPG'  # 본인이 선택한 이미지의 경로에 맞게 바꿔 주세요. 
# img_orig = cv2.imread(img_path) 
print (img_orig.shape)
# %%

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
# %%
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

#%%
plt.figure(figsize=(10,10))
plt.imshow(seg_map)
plt.show()
#%%
for i in seg_map:
    print(i)

# %%
import numpy as np
import pandas as pd
LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
]

def labe(list):
    lab = pd.DataFrame()
    lab['label'] = list
    
    print(lab)
    num = int(input('어떤 대상을 출력하시겠어요?(숫자만 입력해주세요):'))
    return num
n=0
while n==0:
    try:
        num = labe(LABEL_NAMES)
        label_name = LABEL_NAMES[num]
        a = input(label_name+'를 선택하시겠어요?(y/n)')
        if a != 'y':
            continue
        n=1
    except:
        print('적절한 숫자가 아닙니다.')




#%%
seg_map = np.where(seg_map==num,15,0)

plt.imshow(seg_map)
plt.title(label_name+' segmentation map')
#%% 
img_show = img_resized.copy()
# seg_map = np.where(seg_map == 12, 15, 0) # 예측 중 사람만 추출, 위에 선택한 것만
img_mask = seg_map * (255/seg_map.max()) # 255 normalization
img_mask = img_mask.astype(np.uint8)
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.35, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()
# %%
img_mask_up = cv2.resize(img_mask, img_orig.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
_, img_mask_up = cv2.threshold(img_mask_up, 128, 255, cv2.THRESH_BINARY)

ax = plt.subplot(1,2,1)
plt.imshow(img_mask_up, cmap=plt.cm.binary_r)
ax.set_title('Original Size Mask')

ax = plt.subplot(1,2,2)
plt.imshow(img_mask, cmap=plt.cm.binary_r)
ax.set_title('DeepLab Model Mask')

plt.show()

# %%
# 인물모드 배경 focus out
img_orig_blur = cv2.blur(img_orig, (20,20)) #(13,13)은 blurring  kernel size를 뜻합니다. 
plt.imshow(cv2.cvtColor(img_orig_blur, cv2.COLOR_BGR2RGB))
plt.show()
#%%
# 배경 합성시 배경 선택
n=0
while n==0:
    print('배경사진을 골라보겠습니다.')
    num = select_img(image_list)
    image_name = image_list[num-1]
    img_bg = cv2.imread(image_path + image_name)
    img_show_bg = img_bg.copy()
    img_rgb_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)    
    plt.imshow(img_rgb_bg)
    only_imagename = os.path.splitext(image_name)[0]
    plt.title(only_imagename)
    plt.show()
    end=input("이 사진으로 하시겠어요? (y/n)\n\n:")
    if end == 'y' or end == 'yes' or end == 'Y':
        n=1


#%%
# 배경과 자연스럽게 합성할 수 있도록 잘라서 이미지 크기를 조정한다.
def get_croped_image(img_orig,img_mask_up,bg_img,loc='down'):
    x,y,_ = bg_img.shape
    a,b = img_mask_up.shape
    if loc == 'down':
        new_a = b*x//y
        croped_img = img_orig[-new_a:,:,:]
        croped_img_mask = img_mask_up[-new_a:,:]
        croped_img = cv2.resize(croped_img, (y,x))
        croped_img_mask = cv2.resize(croped_img_mask, (y,x))
    return croped_img, croped_img_mask

croped_img, croped_img_mask = get_croped_image(img_orig,img_mask_up,img_bg)

#%%
plt.subplot(121)
plt.imshow(cv2.cvtColor(croped_img, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(croped_img_mask)
plt.show()

# %%
# 인물사진 용
img_mask_color = cv2.cvtColor(img_mask_up, cv2.COLOR_GRAY2BGR)
img_bg_mask = cv2.bitwise_not(img_mask_color)
img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
plt.show()

#%%
# 배경 합성용
# img_bg_mask = cv2.bitwise_not(img_mask_color)
# img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
# plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
# plt.show()
# %%
# 인물사진용
img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()
#%%
# 고양이용

img_concat = np.where(img_mask_color==255, img_orig*2, img_bg_blur)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()
# %%
# 배경합성용
img_mask_color = cv2.cvtColor(croped_img_mask, cv2.COLOR_GRAY2BGR)
# croped_img의 화질 및 색감 변경
croped_img_blur = cv2.blur(croped_img, (10,10)) #(13,13)은 blurring  kernel size를 뜻합니다. 
new_croped_img = croped_img_blur//4 * 3 
# 빼기로 하니 음수값이 이상하게 그려지고, 1.5로 나누면 오류가 난다. 3을 곱하고 4로 나누면 이상하게 더 어둡다
img_concat = np.where(img_mask_color==255, new_croped_img, img_bg)
img_concat_blur = cv2.blur(img_concat, (10,10)) #(13,13)은 blurring  kernel size를 뜻합니다. 
plt.imshow(cv2.cvtColor(img_concat_blur, cv2.COLOR_BGR2RGB))
plt.show()

#%%
# 인물사진용
plt.figure(figsize=(10,20))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
plt.title('original '+only_imagename)
plt.subplot(122)
plt.title('focusing '+only_imagename)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()
#%%
# 배경 합성
plt.figure(figsize=(10,20))
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(img_concat_blur, cv2.COLOR_BGR2RGB))
plt.show()

# %%
cv2.imwrite('output/'+only_imagename+'nice.png',img_concat)

# %%
import os
os.getcwd()