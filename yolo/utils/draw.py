import os
import pickle
import random
import time
import torch
import cv2
import numpy


def rectangle(predictions, image, classes, output_folder, path=None):
    """
    Draw rectangle over objects in image
    :param predictions:  predicted objects
    :param image: original image
    :param classes: object classes
    :param path: path to the image
    """
    for each in predictions:
        colors = pickle.load(open("yolo/data/pallete", "rb"))
        c1 = tuple(each[1:3].int())
        c2 = tuple(each[3:5].int())
        cls = int(each[-1])
        label = f"{classes[cls]}"
        color = random.choice(colors)
        cv2.rectangle(image, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

    output_folder = output_folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    if path:
        image_name = path.split('/')[-1]
    else:

        image_name = f'image_{str(time.time()).split(".")[-1]}.jpg'
    result_path = f"{os.getcwd()}/{output_folder}/output_{image_name}"
    cv2.imwrite(result_path, image)


def get_boxes(predictions):
    """
    Get boxes from predictions
    :param predictions: prediction
    :return: upper left point, bottom right point, label
    """
    boxes = list()
    predictions_list = list()
    # for each in predictions:
    #     is_none = False
    #     for element in each:
    #         print(element)
    #         if element is torch.tensor(data=float('nan')):
    #             is_none = True
    #             break
    #     if not is_none:
    #         predictions_list.append(each)

    for each in predictions:
        try:
            upper_left = (int(each[1]), int(each[2]))
            bottom_right = (int(each[3]), int(each[4]))
            label = int(each[-1])
            boxes.append([upper_left, bottom_right, label])
        except ValueError:
            continue
    return boxes


def rectangle_alpr(predictions, image, args, path=None):
    """
    Draw rectangle over objects in image
    :param predictions:  predicted objects
    :param image: original image
    :param args: program parameters
    :param path: path to the image
    """
    for each in predictions:
        colors = pickle.load(open("yolo/data/pallete", "rb"))
        c1 = (each[0][0], each[0][1])
        c2 = (each[1][0], each[1][1])
        color = random.choice(colors)
        cv2.rectangle(image, c1, c2, color, 3)

    output_folder = args.out_folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    if path:
        image_name = path.split('/')[-1]
    else:

        image_name = f'image_{str(time.time()).split(".")[-1]}.jpg'
    result_path = f"{os.getcwd()}/{output_folder}/output_{image_name}"
    cv2.imwrite(result_path, image)


def normalize_alpr(_car_coordinates, _alpr_coordinates):
    new_coordinates = []
    for result in _alpr_coordinates:
        new_x1 = result[0][0] + _car_coordinates[0][0]
        new_y1 = result[0][1] + _car_coordinates[0][1]
        new_x2 = result[1][0] + _car_coordinates[0][0]
        new_y2 = result[1][1] + _car_coordinates[0][1]
        new_coordinates.append([(new_x1, new_y1), (new_x2, new_y2), result[2]])
    return new_coordinates


def plot_images(_images_folder):
    images_paths = [os.path.join(_images_folder, _image) for _image in os.listdir(_images_folder) if
                    _image.endswith('jpg')]
    random_images = random.sample(range(1, len(images_paths)), 16)
    images = [cv2.imread(images_paths[_index]) for _index in random_images]

    height = sum(image.shape[0] for image in images) // 4
    width = sum(image.shape[1] for image in images) // 4
    output = numpy.zeros((height, width, 3))

    for i_index in range(4):
        for j_index, image in enumerate(images[4 * i_index: (i_index + 1) * 4]):
            h, w, d = image.shape
            output[h * i_index:h * (1 + i_index), w * j_index:w * (j_index + 1)] = image

    cv2.imwrite("results.jpg", output)