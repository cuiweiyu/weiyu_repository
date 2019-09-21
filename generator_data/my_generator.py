import json
import os

import cv2
import numpy as np
from sklearn.utils import shuffle


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


def load_train(train_path, width, height, batch_size):
    classes = np.zeros(61)
    root = os.getcwd()
    with open(train_path, 'r') as load_f:
        load_dict = json.load(load_f)
        start = 0
        end = batch_size
        num_epochs = 0
        while True:
            images = []
            labels = []
            number = np.random.random_integers(0, len(load_dict) - 1, batch_size)
            for image in number:
                index = load_dict[image]["disease_class"]
                path = load_dict[image]['image_id']
                img_path = os.path.join(root, 'new_train', 'images', path)
                image_data = cv_imread(img_path)
                image_data = cv2.resize(image_data, (width, height), 0, 0, cv2.INTER_LINEAR)
                image_data = image_data.astype(np.float32)
                image_data = np.multiply(image_data, 1.0 / 255.0)
                images.append(image_data)
                label = np.zeros(len(classes))
                label[index] = 1
                labels.append(label)
            images = np.array(images)
            labels = np.array(labels)
            yield images, labels


def load_validate(validate_path, width, height):
    root = os.getcwd()
    with open(validate_path, 'r') as load_f:
        load_dict = json.load(load_f)
        # num_image = len(load_dict)
        # 只产生512个数据，避免内存过大
        while True:
            images = []
            labels = []
            classes = np.zeros(61)
            number = np.random.random_integers(0, len(load_dict) - 1, 512)
            for image in number:
                index = load_dict[image]["disease_class"]
                path = load_dict[image]['image_id']
                img_path = os.path.join(root, 'AgriculturalDisease_validationset', 'images', path)
                image_data = cv_imread(img_path)
                image_data = cv2.resize(image_data, (width, height), 0, 0, cv2.INTER_LINEAR)
                image_data = image_data.astype(np.float32)
                image_data = np.multiply(image_data, 1.0 / 255.0)
                images.append(image_data)
                label = np.zeros(len(classes))
                label[index] = 1
                labels.append(label)
            images = np.array(images)
            labels = np.array(labels)
            yield images, labels


times = 3070
batch_size = 64
model.fit_generator(load_data.load_train(train_path, img_rows, img_cols, batch_size=batch_size),
                    steps_per_epoch=times,
                    verbose=1,
                    epochs=100,
                    validation_data=load_data.load_validate(validate_path, img_rows, img_cols),
                    validation_steps=1,
                    shuffle=True)
