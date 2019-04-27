import sys

import cv2

from utils.configuration import Config
from utils.detection_utils import get_car_coordinates, get_alpr_coordinates, get_character_coordinates
from utils.model_utils import car_detector, alpr_detector, characters_detector

image_path = 'images/alpr-1.png'
car_coordinates, image = get_car_coordinates(car_detector=car_detector, image=image_path, config=Config)
if not car_coordinates:
    print('No cars found')
    sys.exit(0)

for car in car_coordinates:
    cv2.rectangle(image,
                  car[0],
                  car[1],
                  color=(56, 12, 42),
                  thickness=2)
    car_image = image[car[0][1]: car[1][1], car[0][0]:car[1][0]]
    alpr_coordinates, _ = get_alpr_coordinates(alpr_detector, car, image, config=Config, first=True)

    cv2.rectangle(car_image,
                  alpr_coordinates[0],
                  alpr_coordinates[1],
                  color=(89, 12, 42),
                  thickness=2)
    alpr_image = car_image[alpr_coordinates[0][1]: alpr_coordinates[1][1],
                 alpr_coordinates[0][0]:alpr_coordinates[1][0]]

    characters_coordinates, alpr_image = get_character_coordinates(characters_detector, alpr_coordinates, alpr_image,
                                                                   config=Config)

    for character_coordinate in characters_coordinates:
        cv2.rectangle(alpr_image,
                      character_coordinate[0],
                      character_coordinate[1],
                      color=(12, 234, 12),
                      thickness=1)
    car_image[alpr_coordinates[0][1]: alpr_coordinates[1][1], alpr_coordinates[0][0]:alpr_coordinates[1][0]] = alpr_image
    image[car[0][1]: car[1][1], car[0][0]:car[1][0]] = car_image


cv2.imshow('TiTle', image)
cv2.waitKey(0)
