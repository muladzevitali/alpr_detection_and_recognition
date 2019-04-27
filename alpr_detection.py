import os
import torch
import time
import cv2
import sys

from characters_recognition import predict
from detection.utils.flags import init_detection_config
from resnet.model import resnet18
from yolo.utils.detector import detection
from yolo.utils.loader import load_model
from yolo.yolo.preprocess import prep_image


def detect_cars(image_path, draw=False):
    """
    Detect car from an image
    :param draw:
    :param image_path: string -- path to the image
    :return: list or None -- car coordinates or None
    """
    car_image_loader = prep_image(image_path, car_detector_height, path=True)
    car_results = detection(car_image_loader, car_detector, flags, draw=draw, detector='cars')
    car_results = list(filter(lambda result: result[2] in CAR_LABELS, car_results))
    if not car_results == 0:
        return car_results, car_image_loader[1]
    return None, None


def detect_alpr_default(_car_coordinates, _image, draw=False):
    """
    Detect alpr from image and car coordinates
    :param draw:
    :param _car_coordinates: list -- e.g [(x0, y0), (x1, y1), 2]
    :param _image: numpy array -- image array
    :return: list or None -- alpr_coordinates or None if not found
    """
    top_left, bottom_right = _car_coordinates[0:2]
    cropped_image = _image[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]]
    alpr_image_loader = prep_image(cropped_image, alpr_detector_height, path=False)
    alpr_coordinates = detection(alpr_image_loader, alpr_detector, flags, draw=draw, detector='alpr')
    if not alpr_coordinates == 0:
        return alpr_coordinates, cropped_image
    return None, None


def detect_alpr(_cars_coordinates, _image, draw=False):
    """
    Detect alpr from many cars on one image
    :param draw:
    :param _cars_coordinates: list -- e.g [[(x0, y0), (x1, y1), 2], [(x0, y0), (x1, y1), 2] ]
    :param _image: numpy array -- image array
    :return:
    """
    if type(_cars_coordinates[0]) == tuple:
        _cars_coordinates = [_cars_coordinates]
    _alprs_coordinates = []
    _car_image = None
    for _car_coordinates in _cars_coordinates:
        alpr_coordinates, _car_image = detect_alpr_default(_car_coordinates, _image, draw)
        if alpr_coordinates:
            _alprs_coordinates.append(alpr_coordinates)
        return _alprs_coordinates, _car_image


def detect_character_default(_alpr_coordinates, _image, draw=False):
    """
    Detect alpr from image and car coordinates
    :param draw:
    :param _alpr_coordinates: list -- e.g [(x0, y0), (x1, y1), 2]
    :param _image: numpy array -- image array
    :return: list or None -- alpr_coordinates or None if not found
    """
    top_left, bottom_right = _alpr_coordinates[0][0:2]
    cropped_image = _image[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]]
    alpr_image_loader = prep_image(cropped_image, character_detector_height, path=False)
    character_coordinates = detection(alpr_image_loader, character_detector, flags, draw=draw, detector='characters')
    if not character_coordinates == 0:
        return character_coordinates, cropped_image
    return None, None


def detect_characters(_alprs_coordinates, _image, draw=False):
    """
    Detect alpr from many cars on one image
    :param draw:
    :param _alprs_coordinates: list -- e.g [[(x0, y0), (x1, y1), 2], [(x0, y0), (x1, y1), 2] ]
    :param _image: numpy array -- image array
    :return:
    """
    if type(_alprs_coordinates[0]) == tuple:
        _alpr_coordinates = [_alprs_coordinates]
    _characters_coordinates = []
    _alpr_image = None
    for _alpr_coordinates in _alprs_coordinates:
        character_coordinates, _alpr_image = detect_character_default(_alpr_coordinates, _image, draw)
        if character_coordinates:
            _characters_coordinates.append(character_coordinates)
    return _characters_coordinates


def recognize_characters(_alprs_coordinates, _characters_coordinates, _image, char_recognition_net):
    """

    :param _alprs_coordinates: list -- e.g [(x0, y0), (x1, y1), 2]
    :param _characters_coordinates: list -- e.g [[(x0, y0), (x1, y1), 2]]
    :param _image:  numpy array -- image array
    :param char_recognition_net:
    :return: string -- extracted text form license plate
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    alpr_top_left, alpr_bottom_right = _alprs_coordinates[0][0][0:2]
    alpr_image = _image[alpr_top_left[1]:alpr_bottom_right[1], alpr_top_left[0]: alpr_bottom_right[0]]
    color = (255, 255, 255)
    _text = ''
    for coordinate in _characters_coordinates[0]:
        top_left, bottom_right = coordinate[0:2]
        cropped_image = alpr_image[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]]
        char = predict(cropped_image, char_recognition_net)
        alpr_image = cv2.rectangle(alpr_image, top_left, bottom_right, color, 1)
        alpr_image = cv2.putText(alpr_image, char, top_left, font, 0.4, color, 1, lineType=cv2.LINE_AA)
        _text += char
    _image = cv2.rectangle(_image, alpr_top_left, alpr_bottom_right, color, 1)
    _image = cv2.putText(_image, _text, alpr_top_left, font, 0.4, color, 1, lineType=cv2.LINE_AA)
    im_name = time.time()
    cv2.imwrite(f'results/result{im_name}.jpg', _image)
    return _text


if __name__ == '__main__':

    # Coco dataset indexes of: Cars, Motorbike, Bus, Truck
    CAR_LABELS = [2, 3, 5, 7]
    # Get default parameters for detection
    flags = init_detection_config()
    # Detector objects

    car_detector = load_model(flags, detector='cars')
    car_detector_height = car_detector.net_info['height']
    # load character recognition network

    char_recognition = resnet18(pretrained=True)
    char_recognition.load_state_dict(torch.load('yolo/weights/characters_recognition.weights', map_location='cpu'))

    alpr_detector = load_model(flags, detector='alpr')
    alpr_detector_height = alpr_detector.net_info['height']

    character_detector = load_model(flags, detector='characters')
    character_detector_height = alpr_detector.net_info['height']

    INPUT_IMAGE = 'images/3.png'
    # Locate the car on the image and get the image back to feed the alpr detector
    cars_coordinates, car_image = detect_cars(INPUT_IMAGE, draw=False)
    for cars_coordinat in cars_coordinates:
        if None in [cars_coordinates]:
            print('No Cars were found')
            sys.exit(0)

        # Locate alpr from car image and get the image back to feed characters detector
        alprs_coordinates, alpr_image = detect_alpr(cars_coordinat, car_image, draw=False)
        if not alprs_coordinates:
            print('No ALPRS found ...')
            sys.exit(0)

        # Locate characters form alpr and get back character coordinates to feed character recognize
        characters_coordinates = detect_characters(alprs_coordinates, alpr_image, draw=False)
        if not characters_coordinates:
            print('No characters found ...')
            sys.exit(0)

        text = recognize_characters(alprs_coordinates, characters_coordinates, alpr_image,char_recognition)
        print(text)
        break
