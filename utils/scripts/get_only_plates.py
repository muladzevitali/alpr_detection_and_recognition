import os

import cv2

PLATES_PATH = '/home/vitali/Projects/dataset/texts'
IMAGES_PATH = '/home/vitali/Projects/dataset/images'


# Done
def get_car_coordinate(_text_file_path):
    with open(_text_file_path) as _input_file:
        _file = _input_file.readlines()
    _car_coordinates = None
    for line in _file:
        if "position_vehicle" in line:
            _car_coordinates = line
    _car_coordinates = _car_coordinates.split()[1:]
    _car_coordinates = [int(each) for each in _car_coordinates]
    _car_coordinates = [(_car_coordinates[0], _car_coordinates[1]),
                        (_car_coordinates[0] + _car_coordinates[2], _car_coordinates[1] + _car_coordinates[3])]

    return _car_coordinates


def get_plate_coordinates(_text_file_path):
    with open(_text_file_path) as _input_file:
        _file = _input_file.readlines()
    _plate_coordinates = None
    for line in _file:
        if "position_plate" in line:
            _plate_coordinates = line
    _plate_coordinates = _plate_coordinates.split()[1:]
    _plate_coordinates = [int(each) for each in _plate_coordinates]
    _plate_coordinates = [(_plate_coordinates[0], _plate_coordinates[1]),
                          (
                          _plate_coordinates[0] + _plate_coordinates[2], _plate_coordinates[1] + _plate_coordinates[3])]

    return _plate_coordinates


if __name__ == "__main__":
    for text_file in os.listdir(PLATES_PATH):
        text_file_path = os.path.join(PLATES_PATH, text_file)
        image_path = os.path.join(IMAGES_PATH, text_file.replace('txt', 'png'))
        car_coordinates = get_car_coordinate(text_file_path)
        plate_coordinates = get_plate_coordinates(text_file_path)

        new_coordinates = [(plate_coordinates[0][0] - car_coordinates[0][0],
                           plate_coordinates[0][1] - car_coordinates[0][1]),
                           (-car_coordinates[0][0] + plate_coordinates[1][0],
                           -car_coordinates[0][1] + plate_coordinates[1][1])]

        image = cv2.imread(image_path)
        cropped_image = image[car_coordinates[0][1]: car_coordinates[1][1], car_coordinates[0][0]:car_coordinates[1][0]]

        new_image_path = os.path.join('/home/vitali/Projects/dataset/plates_over_cars/image', text_file.replace('txt', 'png'))
        cv2.imwrite(new_image_path, cropped_image)

        new_text_path = os.path.join('/home/vitali/Projects/dataset/plates_over_cars/texts', text_file)
        with open(new_text_path, 'w') as _output_file:
            _output_file.write(str(1) + '\n')

            _string = f'{new_coordinates[0][0]} {new_coordinates[0][1]} {new_coordinates[1][0]} {new_coordinates[1][1]}'
            _output_file.write(_string)
        # cv2.rectangle(cropped_image, new_coordinates[0], new_coordinates[1], (213, 123, 12), thickness=10)
        # cv2.imshow('Titpe', cropped_image)
        # cv2.waitKey(0)
        print(new_coordinates)
        # break
