import os
from PIL import Image

def sample(source_root_path,save_root_path,size):
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    label_list = os.listdir(source_root_path)
    for label in label_list:

        save_label_path = os.path.join(save_root_path, label)
        if not os.path.exists(save_label_path):
            os.mkdir(save_label_path)
        source_label_path = os.path.join(source_root_path, label)

        imgs_list = os.listdir(source_label_path)
        imgs_list.sort(key=lambda x: int(x[:-4]))

        imgs_length = len(imgs_list)
        step = imgs_length // size
        for i in range(0, step):
            step_imgs = imgs_list[i * size:size * (i + 1)]
            save_step_path = os.path.join(save_label_path, str(i))
            if not os.path.exists(save_step_path):
                os.mkdir(save_step_path)
            for step_img in step_imgs:
                save_img_path = os.path.join(save_step_path, step_img)
                source_img_path = os.path.join(source_label_path, step_img)
                img = Image.open(source_img_path)
                img.save(save_img_path)

if __name__ == '__main__':
    source_root_path = "/home/xavier/data/dvs_kmeans"
    source_root_path_list = os.listdir(source_root_path)
    save_root_path = r"/home/xavier/data/DvsGesture_frame_kmeans"
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    for source_path in source_root_path_list:
        save_path = os.path.join(save_root_path,source_path)
        source_path = os.path.join(source_root_path,source_path)
        sample(source_path,save_path,16)
