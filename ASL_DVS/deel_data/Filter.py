import os
from PIL import Image
import numpy as np

def filter(img, k_size, n_features, background_color):
    step = k_size // 2
    thread = 1.0 * n_features / k_size ** 2

    for i in range(step, img.shape[0] - step, k_size):
        for j in range(step, img.shape[1] - step, k_size):
            fitter_kerize = img[i - step:i + step:, j - step:j + step]

            acc = fitter_kerize == background_color

            if (acc.mean() > thread):
                img[i - step:i + step, j - step:j + step] = background_color
    return img
def creat_filter(source_root_path,save_root_path):
    source_path_list = os.listdir(source_root_path)
    for source_path in source_path_list:
        source_labels_list = os.listdir(os.path.join(source_root_path,source_path))
        save_path = os.path.join(save_root_path,source_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for source_label in source_labels_list:
            save_labels = os.path.join(save_path,source_label)
            if not os.path.exists(save_labels):
                os.mkdir(save_labels)
            counts = os.listdir(os.path.join(source_root_path,source_path,source_label))
            for count in counts:
                count_save_path = os.path.join(save_labels,count)
                if not os.path.exists(count_save_path):
                    os.mkdir(count_save_path)
                imgs = os.listdir(os.path.join(source_root_path,source_path,source_label,count))
                for img in imgs:
                    save_image_path = os.path.join(count_save_path, img)
                    print(save_image_path)
                    img = Image.open(os.path.join(source_root_path,source_path,source_label,count,img))
                    img = Image.fromarray(filter(np.array(img),10,9,2))
                    img.save(save_image_path)
if __name__ == '__main__':
    source_root_path = "/home/xavier/data/DvsGesture_frame"
    save_root_path = "/home/xavier/data/DvsGesture_frame_filter"
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    creat_filter(source_root_path,save_root_path)