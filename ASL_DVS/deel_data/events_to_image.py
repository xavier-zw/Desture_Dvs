from aedat_to_events import events_from_aedat,img_from_events_stepclip,img_from_events_timeclip,using_Kmeans_events_to_img,three_channle_img
import os
import glob
import re
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    path = "/home/xavier/data"
    save_path = "/home/xavier/data/dvs_3channle"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    source_paths = glob.glob(path+r'/DvsGesture/user*.aedat')
    for source_path in tqdm(source_paths):
        name = re.search(r"user.*", source_path).group()[:-6]
        label_path = os.path.join(path,"DvsGesture/"+name+"_labels.csv")

        events = events_from_aedat(source_path)
        labels = pd.read_csv(label_path)
        for label in labels.itertuples():
            class_ = label[1]
            start_time = label[2]
            end_time = label[3]
            temp = os.path.join(save_path,name+r"/"+str(class_))
            if not os.path.exists(temp):
                os.makedirs(temp)
            three_channle_img(temp, events[(events[:,3]>=start_time) & (events[:,3]<= end_time)], 128, 128, 4000)
