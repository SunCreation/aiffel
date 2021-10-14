#%%
import pickle
import numpy as np
# https://github.com/ageitgey/face_recognition
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import face_recognition as face_rec

test_dir = 'test'
test_list = os.listdir(test_dir)
test_list

def get_cropped_face(image_file):
    image = face_rec.load_image_file(image_file)
    face_locations = face_rec.face_locations(image) 
    cropped_face = None
    if len(face_locations) != 0 :
        a,b,c,d = face_locations[0]
        cropped_face = image[a:c,d:b,:]
    return cropped_face


def get_face_embedding(face):
    return face_rec.face_encodings(face)

def get_embedding_dict(dir_path):
    file_list = os.listdir(dir_path)
    embedding_dict = {}

    for file in file_list:
        filename = '.'.join(file.split('.')[:-1]) # os.path.splitext(file)[0]
        image_file = os.path.join(dir_path, file)
        face= get_cropped_face(image_file)
        if face is None : continue

        try:
            embedding_dict[filename]=get_face_embedding(face)[0]
        except:
            pass
    
    return embedding_dict

test_dict = get_embedding_dict(test_dir)

my_dict = pickle.load(open('my_model.p', 'rb'))


def get_distance(test, name):
    return np.linalg.norm(test_dict[test] - my_dict[name], ord=2)


def get_sort_key_func(name1):
    def get_distance_from_name1(name2):
        return get_distance(name1, name2)
    return get_distance_from_name1


def get_nearest_face(name, top=5):
    sort_key_func = get_sort_key_func(name)
    nearlist = [i for i, j in sorted(my_dict.items(), key=lambda x:sort_key_func(x[0]))[:top]]
    print("닮은 꼴 연예인 순위 발표!!")
    n = 1
    for i in nearlist:
        print(str(n) + "위 :", i + ",", "- 얼굴 거리: ", sort_key_func(i))
        n += 1
    
    print(nearlist[0],"를 가장 닮았네요:)", sep='')
    return nearlist
#%%

get_nearest_face('조해창',10)