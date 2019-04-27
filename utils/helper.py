import cv2
import numpy as np


def load_image(data):
    """
    Load image from request.data object
    :param data: request.data object
    :return: numpy, grayscale image
    """
    image_array = np.fromstring(data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
