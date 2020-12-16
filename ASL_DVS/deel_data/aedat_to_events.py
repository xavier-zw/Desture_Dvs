import numpy as np
import struct
from PIL import Image
import os
from sklearn.cluster import KMeans
def get_file(path): #读取文件且将指针移动到第一个有效数据字节
    f = open(path,"rb")
    count = 0
    while (1):
        line = f.readline()
        if line[0] != 35:
            break
        count += 1
    f.close()
    f = open(path,"rb")
    for i in range(count):
        f.readline()
    return f

def re_data(x):
    res = ""
    flag = True
    for i in x:
        i = bin(i)[2:]
        res = i.rjust(8,"0") + res
    if(res[63]=='0'):
        flag = False
    p = int(res[62:63],2)
    x = int(res[32:47],2)
    y = int(res[47:62],2)
    t = int(res[:32],2)
    return flag,p,x,y,t

def get_events(f): #获取事件信息
    res = []
    while (1):
        try:
            head = f.read(28)
            heads = [x for x in struct.unpack("2H6I", head)]
            # print(heads)
            for i in range(heads[6]):
                flag, p, x, y, t = re_data(f.read(8))
                if (flag):
                    res.append([p, x, y, t])
        except:
            break
    f.close()
    return res
def events_from_aedat(path): #读取事件成（p,x,y,t)
    f = get_file(path)
    events = get_events(f)
    events = np.array(events)
    return events

def img_from_events_stepclip(save_path,events,width,height,step): #将事件按记件转化为img
    length = int(len(events)/step)
    for count in range(length):
        img = np.full((width, height), 2)
        for p, x, y, t in events[step*count:step*(count+1)]:
            img[x][y] = p
        img = Image.fromarray(np.uint8(img))
        img = img.rotate(-90)
        img.save(os.path.join(save_path, str(count).rjust(len(str(length)), "0") + ".png"))

def img_from_events_timeclip(save_path,events,width,height,step): #将事件按时间片转化为img

    length = int((events[-1][3] - events[0][3])/step)
    start_time = events[0][3]
    for count in range(length):
        img = np.full((width, height), 2)
        for p, x, y, t in events[(events[:,3]>=start_time+count*step) & (events[:,3]< start_time + (count+1)*step)]:
            img[x][y] = p
        img = Image.fromarray(np.uint8(img))
        img = img.rotate(-90)
        img.save(os.path.join(save_path, str(count).rjust(len(str(length)), "0") + ".png"))
    print("Save success!")

def using_Kmeans_events_to_img(save_path,events,width,height,step):
    length = int(len(events)/step)
    x = np.array(events[:,1:])
    model = KMeans(n_clusters=length).fit_predict(x)
    _,index = np.unique(model,return_index=True)
    index.sort()
    for i in range(length):
        img = np.full((width, height), 2)
        temp = events[index[i]:index[i+1]] if i != length-1 else events[index[i]:]
        for p, x, y, t in temp:
            img[x][y] = p
        img = Image.fromarray(np.uint8(img))
        img = img.rotate(-90)
        img.save(os.path.join(save_path, str(i).rjust(len(str(length)), "0") + ".png"))


def three_channle_img(save_path, events, width, height, step):
    length = int(len(events) / step)
    for count in range(length):
        img = np.full((width, height, 3), 127)
        img_on = np.full((width, height), 127)
        img_off = np.full((width, height), 127)
        img_heatmap = np.full((width, height), 127)
        for p, x, y, t in events[step * count:step * (count + 1)]:
            p = int(p)
            if p == 1:
                img_on[x][y] += 1
            else:
                p = -1
                img_off[x][y] -= 1
            img_heatmap[x][y] += p
        img[:, :, 0] = img_on
        img[:, :, 1] = img_off
        img[:, :, 2] = img_heatmap
        img = Image.fromarray(np.uint8(img))
        img = img.rotate(-90)

        img.save(os.path.join(save_path, str(count).rjust(len(str(length)), "0") + ".png"))
if __name__ == '__main__':
    pass