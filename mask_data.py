import os, sys
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np 
import cv2 
import multiprocessing


rootpath = "/data/leo_data/CASIA-B" 
path1 = "/data/leo_data/CASIA-B-Mask-Leg/"
path2 = ""
T_H = 64
T_W = 64

def cut_img(path):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    img = Image.open(path)
    img = np.array(img).astype("uint8")
    if img.sum() <= 10000:
        return None
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]

    return img.astype('uint8')

def MkDir():
    id_dirs = os.listdir(rootpath)
    for id_dir in id_dirs: 
        id_path = path1 + str(id_dir) +"/" 
        if not os.path.exists(id_path):
            os.mkdir(id_path) 
        path2 = rootpath + "/" + id_dir + "/"
        state_dirs = os.listdir(path2)

        for state_dir in state_dirs: 
            state_path = id_path + str(state_dir) + "/"
            if not os.path.exists(state_path):
                os.mkdir(state_path)
            path2 = rootpath + "/" + id_dir + "/" + state_dir + "/"
            view_dirs = os.listdir (path2)

            for view_dir in view_dirs:
                view_path = state_path + str(view_dir) + "/"
            
                if not os.path.exists(view_path):
                    os.mkdir(view_path)

                path2 = rootpath + "/" + id_dir + "/" + state_dir + "/" + view_dir + "/"
                in_image_dirs = os.listdir (path2)

                for in_image_dir in in_image_dirs:
                    inpath = path2 + in_image_dir

                    # img = Image.open(inpath)
                    # arry_img = np.asarray(img)
                    # img_cv = cv2.imread(inpath)
                    # plt.imshow(img)
                    # img2 = img.crop((0,0,80,80))  #裁剪原图中一部分作为覆盖图片
                    # img2 = Image.open(inpath)
                    # draw = ImageDraw.Draw(img2)
                    # draw.rectangle((140,40,180,70), fill = 0)
                    out_image_path = view_path + str(in_image_dir)
                    if not os.path.exists(out_image_path):     
                        
                        img = cut_img(inpath)
                        if img is not None: 
                            img = Image.fromarray(img.astype('uint8')).convert('RGB')
                            draw = ImageDraw.Draw(img)
                            draw.rectangle((0,38,64,64), fill = 0)
                            img.save(out_image_path)
            # print(state_path)
if not os.path.exists(path1):
    os.mkdir(path1)
MkDir() 

