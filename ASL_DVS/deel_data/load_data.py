from PIL import Image
import os
import torch
import numpy as np

def load_data(paths,size):
    res_x,res_y = [],[]
    for path,y in paths:
        img_list = [str(x) for x in os.listdir(path)]
        img_list.sort(key=lambda x: int(x[:-4]))
        imgs = []
        for i in range(size):
            img = Image.open(os.path.join(path,img_list[i]))
            img = np.array(img)
            imgs.append(torch.from_numpy(img))
        imgs = torch.stack(imgs,dim=0)
        res_x.append(imgs)
        res_y.append(int(y)-1)
    res_x = torch.stack(res_x)
    res_y = torch.LongTensor(np.array(res_y))
    return res_x,res_y

def load_data_one(path,size):
    img_list = [ str(x)[2:-1] for x in os.listdir(path)]
    img_list.sort(key=lambda x: int(x[:-4]))
    imgs = []
    for i in range(size):
        img = Image.open(os.path.join(path,img_list[i]))
        img = np.array(img)
        imgs.append(torch.from_numpy(img))
    imgs = torch.stack(imgs,dim=0)
    return imgs.unsqueeze(0)