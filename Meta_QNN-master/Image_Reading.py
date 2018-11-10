import os
import numpy as np
from PIL import Image

def Image_Reading(height, width, train_wd ,test_wd):
    label_dic = {
        'bicycle' : 0,
        'horse' : 1,
        'ship' : 2,
        'truck' : 3
    }

    #trainset
    image = []
    labels=[]
    height=32
    width=32
    resizing = (height, width)
    train_wd = "C:\\Users\\CSH\\Desktop\\image_example\\train\\"
    for label in label_dic:
        image_dir = train_wd + label
        labeled_image = []
        for path, dir, files in os.walk(image_dir):
            for file in files:
                image_dir = path + '/' + file
                img = Image.open(image_dir)
                img = img.resize(resizing)
                if not img.format == "RGB": # 이미지의 포맷이 RGB가 아닐 경우, RGB로 convert 시킴
                    img = img.convert("RGB")
                image.append(np.array(img))
                labels.append(label_dic[label])

    image=np.array(image)
    labels = np.array(labels)

    from collections import Counter
    Counter(labels)

    n_class = len(np.unique(labels))

    # testset
    test_image = []
    test_labels=[]
    height=32
    width=32
    resizing = (height, width)
    test_wd = "C:\\Users\\CSH\\Desktop\\image_example\\test\\"
    for label in label_dic:
        image_dir = test_wd + label
        labeled_image = []
        for path, dir, files in os.walk(image_dir):
            for file in files:
                image_dir = path + '/' + file
                img = Image.open(image_dir)
                img = img.resize(resizing)
                if not img.format == "RGB": # 이미지의 포맷이 RGB가 아닐 경우, RGB로 convert 시킴
                    img = img.convert("RGB")
                test_image.append(np.array(img))
                test_labels.append(label_dic[label])

    print(np.shape(labels))
    print(np.shape(image))
    print(np.shape(test_labels))
    print(np.shape(test_image))

    return(image, labels, test_image, test_labels, n_class)