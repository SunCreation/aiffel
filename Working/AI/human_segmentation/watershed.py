#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image

# url = 'https://opencv-python.readthedocs.io/en/latest/_images/image021.jpg'

# # 인터넷에서 이미지를 가져온다. 위 처럼 주소를 명확히 한다.
# im = Image.open(requests.get(url, stream=True).raw)
# plt.imshow(im)
# plt.show()

def remarking(img):

    # #%%
    # # 사진으로 받은 파일을 np.array로 바꾸기 위해....... 저장해서 다시 불러온다. 그래야하나??
    # im.save("tmp.jpg")

    # img = cv2.imread("tmp.jpg")

    # os.remove("tmp.jpg")

    #%%
    # plt.show()
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # plt.imshow(gray)
    # print(gray)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # print(ret) == 162.0
    # plt.imshow(thresh)
    # print(cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 궁금해서 쳐봄 == 9


    # Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
    # plt.imshow(opening)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # plt.imshow(sure_bg)
    plt.show()
    # distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
    # 즉, 중심으로 부터 점점 옅어져 가는 영상.
    # 그 결과에 thresh를 이용하여 확실한 FG를 파악
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    # plt.imshow(sure_fg)
    print(ret)

    # Background에서 Foregrand를 제외한 영역을 Unknow영역으로 파악
    unknown = cv2.subtract(sure_bg, sure_fg)
    # plt.imshow(unknown)

    # FG에 라벨링 작업
    ret, markers = cv2.connectedComponents(sure_fg)
    # for i in markers:
    #     print(i) 출력해보면, 각 동전자리에, 숫자가 매겨져 있는데, 연결되지 않은 부분을 카운팅한다.
    markers += 1
    markers[unknown == 255] = 0
    # plt.imshow(markers)

    # watershed를 적용하고 경계 영역에 색 지정
    markers = cv2.watershed(img,markers)

    # for i in markers:
    #     print(i)

    # plt.imshow(markers)


    img[markers == -1] = [255,0,0]

    return img, markers
    # for i in img:
    #     print(i)
    #%%
    # plt.imshow(img) # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# %%
img = cv2.imread('images/겹친사람.JPG')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#%%
for i in thresh:
    print(i)
#%%
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

sure_bg = cv2.dilate(opening,kernel,iterations=3)

plt.show()

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)

print(ret)

unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)

markers += 1
markers[unknown == 255] = 0
plt.imshow(markers)
plt.show

markers = cv2.watershed(img,markers)

img[markers == -1] = [255,0,0]