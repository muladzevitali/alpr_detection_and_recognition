import os

import cv2

_OK = 'ok'


def get_coordinates(test_file_path):
    with open(test_file_path, 'r') as _input_file:
        _rows = _input_file.readlines()[1:]
    _coordinates = []

    for row in _rows:
        _row_coordinates = [int(_each.strip()) for _each in row.split(' ')]
        _coordinates.append(_row_coordinates)
    return _coordinates


def check_if_inside(_car_coordinates, _plate_coordinates):
    if _car_coordinates[0][0] > _plate_coordinates[0]:
        return False
    elif _car_coordinates[0][1] > _plate_coordinates[1]:
        return False
    elif _car_coordinates[0][2] < _plate_coordinates[2]:
        return False
    elif _car_coordinates[0][3] < _plate_coordinates[3]:
        return False
    else:
        return True


if __name__ == '__main__':
    # single_image('images/cars-1.jpg')
    IMAGES_PATH = '/home/vitali/Downloads/characters_bbox/Images/001'
    LABELS_PATH = '/home/vitali/Downloads/characters_bbox/Labels/001'
    images = os.listdir(IMAGES_PATH)
    for _index, image in enumerate(images):
        print(f'{_index} / {len(images)}')
        image_path = os.path.join(IMAGES_PATH, image)
        plate_file_path = image_path.replace('Images', 'Labels').replace('characters_bbox', 'plates_bbox').replace(
            'png', 'txt')
        characters_file_path = plate_file_path.replace('plates_bbox', 'characters_bbox')

        text_file_name = plate_file_path.split('/')[-1]
        frame = cv2.imread(image_path)
        plate_numbers_coordinates = get_coordinates(plate_file_path)
        characters_coordinates = get_coordinates(characters_file_path)
        top_left, bottom_right = plate_numbers_coordinates[0][0:2], plate_numbers_coordinates[0][2:4]

        cropped_image = frame[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]]
        coordinates = []
        number_of_characters = 0
        for character_coordinates in characters_coordinates:
            new_coordinates = []
            if check_if_inside(plate_numbers_coordinates, character_coordinates):
                number_of_characters += 1

                new_coordinates = [character_coordinates[0] - plate_numbers_coordinates[0][0],
                                   character_coordinates[1] - plate_numbers_coordinates[0][1],
                                   -plate_numbers_coordinates[0][0] + character_coordinates[2],
                                   -plate_numbers_coordinates[0][1] + character_coordinates[3]]
                coordinates.append(new_coordinates)

        if number_of_characters > 0:
            _image_new_path = os.path.join('/home/vitali/Downloads/charaplates/Images/001', image)
            cv2.imwrite(_image_new_path, cropped_image)
            _text_file_new_path = os.path.join('/home/vitali/Downloads/charaplates/Labels/001', text_file_name)
            with open(_text_file_new_path, 'w') as _output_file:
                _output_file.write(str(number_of_characters) + '\n')
                for each in coordinates:
                    _string = f'{each[0]} {each[1]} {each[2]} {each[3]}'
                    _output_file.write(_string + '\n')
        # break
