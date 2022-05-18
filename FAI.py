#%%
from PIL import Image
import os, glob

Workingpath = input("\n\n\n\n\n사용자의 home 디렉토리에서 Working파일 사이의 경로를 입력하세요.(ex: /aiffel/assignment ) :")

def resize_images(img_path, nb=0):
    # 가위
    imagesr=glob.glob(img_path + "/rock/*.jpg")  
    
    print(len(imagesr), " images to be resized.")

    # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
    target_size=(28,28)
    for img in imagesr:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img, "JPEG")
        nb +=1

    print(len(imagesr), " images resized.")
    # 바위
    imagess=glob.glob(img_path + "/scissor/*.jpg")  
    print(len(imagess), " images to be resized.")

    # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
    target_size=(28,28)
    for img in imagess:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img, "JPEG")
        nb +=1
        
    print(len(imagess), " images resized.")
    
    # 보
    imagesp=glob.glob(img_path + "/paper/*.jpg")  
    print(len(imagesp), " images to be resized.")

    # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
    target_size=(28,28)
    for img in imagesp:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img, "JPEG")
        nb +=1

    print(len(imagesp), " images resized.")
    return nb

# 가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getenv("HOME") + Workingpath + "/Working/AI/first_AI/"
resize_images(image_dir_path + "mydata/") # 내거
resize_images(image_dir_path + "more/") # 세영님
resize_images(image_dir_path + "moremore/") # 수영님
resize_images(image_dir_path + "train/") # 태균님
resize_images(image_dir_path + "feldata/") # 아이펠거
resize_images(image_dir_path + "up/") # 강민님
print("이미지 resize 완료!")


import numpy as np

def load_data(img_path, number_of_data=1827):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'mydata/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1
    for file in glob.iglob(img_path+'more/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'mydata/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  

    for file in glob.iglob(img_path+'more/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1    

    for file in glob.iglob(img_path+'mydata/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1

    for file in glob.iglob(img_path+'more/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1

    for file in glob.iglob(img_path+'moremore/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'moremore/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1

    for file in glob.iglob(img_path+'moremore/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    for file in glob.iglob(img_path+'train/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'train/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1

    for file in glob.iglob(img_path+'train/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1

    for file in glob.iglob(img_path+'feldata/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'feldata/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1

    for file in glob.iglob(img_path+'feldata/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1

    for file in glob.iglob(img_path+'up/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'up/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1

    for file in glob.iglob(img_path+'up/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1

    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
    return imgs, labels
# 위 함수 이용하여, 데이터 저장 및 정규화

image_dir_path = os.getenv("HOME") + Workingpath + "/Working/AI/first_AI/"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))


import matplotlib.pyplot as plt
plt.imshow(x_train[150])
print('라벨: ', y_train[150])
plt.show()


#%%

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import getenv
getenv("HOME")
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.25,
        height_shift_range=0.25,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True
        #fill_mode='nearest'
        )
datagen2 = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True
        )
# x_train_gen={} # 이거 자체를 np.ndarray로 바꾸는 실험 시작
# A={} #  다시 
# for i in range(1827):
#     x_train_gen[i]= next(iter(datagen.flow(x_train[tf.newaxis,i])))
#     A[i] = x_train_gen[i].astype(np.int64) # 태균님 성과물
# print(x_train[3])
# print(x_train_gen[0])
x_train_gen1, y_train_gen1= next(iter(datagen.flow(x_train,y_train, batch_size=1827)))
x_train_gen2, y_train_gen2= next(iter(datagen.flow(x_train,y_train, batch_size=1827)))
x_train_gen3, y_train_gen3= next(iter(datagen.flow(x_train,y_train, batch_size=1827)))
x_train_gen4, y_train_gen4= next(iter(datagen.flow(x_train,y_train, batch_size=1827)))
x_train_gen5, y_train_gen5= next(iter(datagen2.flow(x_train,y_train, batch_size=1827)))

A = x_train_gen1.astype(np.int64) # 태균님 성과물
B = x_train_gen2.astype(np.int64)
C = x_train_gen3.astype(np.int64)
D = x_train_gen4.astype(np.int64)
E = x_train_gen5.astype(np.int64)

print(x_train.shape)
print(A.shape)
#print(x_train_gen.shape)
plt.figure(figsize= (10,10))
plt.subplot(161)
plt.imshow(x_train[110]) # (1827, 28, 28, 3) 
plt.title("Origin")
print(y_train[110])

plt.subplot(162)
plt.imshow(A[110]) # (1,28,28,3)
plt.title("rota10")
print(y_train_gen1[110])

plt.subplot(163)
plt.imshow(B[314]) # (1,28,28,3)
plt.title("rota10")
print(y_train_gen2[314])

plt.subplot(164)
plt.imshow(C[513]) # (1,28,28,3)
plt.title("rota10")
print(y_train_gen3[513])

plt.subplot(165)
plt.imshow(D[314]) # (1,28,28,3)
plt.title("rota10")
print(y_train_gen4[314])

plt.subplot(166)
plt.imshow(E[314]) # (1,28,28,3)
plt.title("rota15")
print(y_train_gen5[314])
plt.show()
#print(x_train_gen[0])
# val_gen  = train_datagen.flow_from_directory('/content/drive/MyDrive/Data/test',
#                                                  target_size = (28 , 28),
#                                                  batch_size = 32,
#                                                  class_mode = 'categorical',subset='validation')


# type(np.array(A))
# # type(x_train)
# np.array(A)



import tensorflow as tf
from tensorflow import keras
import numpy as np

# model을 직접 만들어 보세요.
# Hint! model의 입력/출력부에 특히 유의해 주세요. 가위바위보 데이터셋은 MNIST 데이터셋과 어떤 점이 달라졌나요?
# Min-Max scaling
x_train_norm = x_train / 255.0
x_train_gen1_norm = A / 255.0
x_train_gen2_norm = B / 255.0
x_train_gen3_norm = C / 255.0
x_train_gen4_norm = D / 255.0
x_train_gen5_norm = E / 255.0

print('최소값:',np.min(x_train_norm), ' 최대값:',np.max(x_train_norm))

#%%

def model_setting(mdl,a,b,c,d):
    #mdl=keras.models.Sequential() # np.ndarray().shape -> (2,3,4)
    #mdl.add(keras.layers.Conv2D(t, (3,3), activation='relu', input_shape=(28,28,3)))
    #mdl.add(keras.layers.MaxPool2D(2,2))
    mdl.add(keras.layers.Conv2D(a, (3,3), activation='elu', input_shape=(28,28,3)))
    mdl.add(keras.layers.MaxPool2D(2,2))
    mdl.add(keras.layers.Conv2D(b, (3,3), activation='elu'))
    mdl.add(keras.layers.MaxPooling2D((2,2)))
    mdl.add(keras.layers.Flatten())
    mdl.add(keras.layers.Dense(c, activation='elu'))  # 시그모이드..?
    mdl.add(keras.layers.Dense(d, activation='softmax')) # 소프트맥스

    print('Model에 추가된 Layer 개수: ', len(mdl.layers))
 
    mdl.summary()
    return mdl

model=keras.models.Sequential()
#model_setting(model,60, 180,540,72,3)
model_setting(model,60,210,72,3)

print("Before Reshape - x_train_norm shape: {}".format(x_train_norm.shape))
x_train_reshaped=x_train_norm #.reshape( -1, 28, 28, 3)  # 데이터갯수에 -1을 쓰면 reshape시 자동계산됩니다.
x_train_gen1_reshaped = x_train_gen1_norm #.reshape(-1,28,28,3)
x_train_gen2_reshaped = x_train_gen2_norm #.reshape(-1,28,28,3)
x_train_gen3_reshaped = x_train_gen3_norm #.reshape(-1,28,28,3)
x_train_gen4_reshaped = x_train_gen4_norm #.reshape(-1,28,28,3)
x_train_gen5_reshaped = x_train_gen5_norm #.reshape(-1,28,28,3)

print("After Reshape - x_train_reshaped shape: {}".format(x_train_reshaped.shape))
print("After Reshape - x_train_gen_reshaped shape: {}".format(x_train_gen1_reshaped.shape))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

#model.fit(x_train_reshaped, y_train, epochs=3)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)

model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)
model.fit(x_train_gen1_reshaped, y_train_gen1, epochs=1)
model.fit(x_train_reshaped, y_train, epochs=1)
model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=1)
model.fit(x_train_gen3_reshaped, y_train_gen3, epochs=1)
model.fit(x_train_gen4_reshaped, y_train_gen4, epochs=1)
model.fit(x_train_gen5_reshaped, y_train_gen5, epochs=1)



# model.fit(x_train_gen_reshaped, y_train_gen, epochs=3)
# model.fit(x_train_reshaped, y_train, epochs=2)
# model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=3)
# model.fit(x_train_gen_reshaped, y_train_gen, epochs=3)
# model.fit(x_train_reshaped, y_train, epochs=2)
# model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=3)
# model.fit(x_train_gen_reshaped, y_train_gen, epochs=3)
# model.fit(x_train_reshaped, y_train, epochs=2)
# model.fit(x_train_gen2_reshaped, y_train_gen2, epochs=4)
# model.fit(x_train_gen_reshaped, y_train_gen, epochs=4)
# model.fit(x_train_gen_reshaped, y_train_gen, epochs=7)



#%%
def load_Tdata(img_path, number_of_data=300):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  

    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1

    print("실험데이터(x_test)의 이미지 개수는", idx,"입니다.")
    return imgs, labels

# 화면출력 놀이
num = 0
while num == 0:
    try:
        INPUT = int(input('''\n\n사용하고 싶은 테스트를 선택하시오.
            
            1. test
            2. real test (It's small size.)
            3. 직접 입력


        Enter your Number: '''))
        if INPUT != 1 and INPUT != 2 and INPUT !=3: 
            print("\n주어진 숫자를 넣어주세요 :)\n\n\n\n\n")
            continue
        if INPUT == 1:
            test_image_dir_path=image_dir_path + "test"
        elif INPUT == 2:
            test_image_dir_path=image_dir_path + "test_real"
        else:
            test_image_dir_path=image_dir_path + input("Enter test_image_folder_name: ")
        a = 0
        a = resize_images(test_image_dir_path, a)
        print(a)
        (x_test, y_test)=load_Tdata(test_image_dir_path, a)
        if sum(sum(sum(sum(x_test)))) == 0 :
            print("파일이 없습니다.\n\n\n\n\n")
            continue
        # 실험데이터 정규화 및 reshape
        x_test_norm = x_test / 255.0
        print("Before Reshape - x_test_norm shape: {}".format(x_test_norm.shape))
        x_test_reshaped=x_test_norm.reshape( -1, 28, 28, 3)
        print("After Reshape - x_test_reshaped shape: {}".format(x_test_reshaped.shape))
        
        # 테스트
        test_loss, test_accuracy = model.evaluate(x_test_reshaped,y_test, verbose=2)
        print("test_loss: {} ".format(test_loss))
        print("test_accuracy: {}".format(test_accuracy))


        predicted_result = model.predict(x_test_reshaped)  # model이 추론한 확률값. 
        predicted_labels = np.argmax(predicted_result, axis=1)

        idx=9  #1번째 x_test를 살펴보자. 
        print('model.predict() 결과 : ', predicted_result[idx])
        print('model이 추론한 가장 가능성이 높은 결과 : ', predicted_labels[idx])
        print('실제 데이터의 라벨 : ', y_test[idx])

        plt.imshow(x_test[idx],cmap=plt.cm.binary)
        plt.show()



        import random
        wrong_predict_list=[]
        for i, _ in enumerate(predicted_labels):
            # i번째 test_labels과 y_test이 다른 경우만 모아 봅시다. 
            if predicted_labels[i] != y_test[i]:
                wrong_predict_list.append(i)
        print(len(predicted_labels))
        print(len(wrong_predict_list))
        #%%
        # wrong_predict_list 에서 랜덤하게 5개만 뽑아봅시다.
        samples = random.choices(population=wrong_predict_list, k=5)

        for n in samples:
            print("예측확률분포: " + str(predicted_result[n]))
            print("라벨: " + str(y_test[n]) + ", 예측결과: " + str(predicted_labels[n]))
            plt.imshow(x_test[n], cmap=plt.cm.binary)
            plt.show()

        end=input("테스트를 마치시겠습니까? (y/n)\n\n:")
        if end == 'y' or end == 'yes' or end == 'Y':
            num = INPUT
    except:
        print("\n숫자만 넣어주세요!\n\n")

# print(test_image_dir_path, "테스트 실행")

# if num == 1:
#     test_image_dir_path=os.getenv("HOME") + "/Working/AI/first_AI/test"
# elif num == 2:
#     test_image_dir_path=os.getenv("HOME") + "/Working/AI/first_AI/test_real"
# else:
#     test_image_dir_path=os.getenv("HOME") + input("Enter test_image_dir_path: 'Home' +")


# resize_images(test_image_dir_path)
# (x_test, y_test)=load_Tdata(test_image_dir_path)




#%%

# # 실험데이터 정규화 및 reshape
# x_test_norm = x_test / 255.0
# print("Before Reshape - x_test_norm shape: {}".format(x_test_norm.shape))
# x_test_reshaped=x_test_norm.reshape( -1, 28, 28, 3)
# print("After Reshape - x_test_reshaped shape: {}".format(x_test_reshaped.shape))
# #%%
# # 테스트
# test_loss, test_accuracy = model.evaluate(x_test_reshaped,y_test, verbose=2)
# print("test_loss: {} ".format(test_loss))
# print("test_accuracy: {}".format(test_accuracy))


# #%%

# predicted_result = model.predict(x_test_reshaped)  # model이 추론한 확률값. 
# predicted_labels = np.argmax(predicted_result, axis=1)

# idx=9  #1번째 x_test를 살펴보자. 
# print('model.predict() 결과 : ', predicted_result[idx])
# print('model이 추론한 가장 가능성이 높은 결과 : ', predicted_labels[idx])
# print('실제 데이터의 라벨 : ', y_test[idx])

# plt.imshow(x_test[idx],cmap=plt.cm.binary)
# plt.show()


# #%%


# import random
# wrong_predict_list=[]
# for i, _ in enumerate(predicted_labels):
#     # i번째 test_labels과 y_test이 다른 경우만 모아 봅시다. 
#     if predicted_labels[i] != y_test[i]:
#         wrong_predict_list.append(i)
# print(len(predicted_labels))
# print(len(wrong_predict_list))
# #%%
# # wrong_predict_list 에서 랜덤하게 5개만 뽑아봅시다.
# samples = random.choices(population=wrong_predict_list, k=5)

# for n in samples:
#     print("예측확률분포: " + str(predicted_result[n]))
#     print("라벨: " + str(y_test[n]) + ", 예측결과: " + str(predicted_labels[n]))
#     plt.imshow(x_test[n], cmap=plt.cm.binary)
#     plt.show()
