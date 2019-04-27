import os

import cv2

PLATES_PATH = '/home/vitali/Projects/dataset/texts'
IMAGES_PATH = '/home/vitali/Projects/dataset/images'


# Done
def get_character_coordinates(_text_file_path):
    with open(_text_file_path) as _input_file:
        _file = _input_file.readlines()
    _car_coordinates = list()
    for line in _file:
        if "char " in line:
            _coordinates = line.split()[2:]
            _coordinates = [int(each) for each in _coordinates]
            _coordinates = [(_coordinates[0], _coordinates[1]),
                            (_coordinates[0] + _coordinates[2], _coordinates[1] + _coordinates[3])]
            _car_coordinates.append(_coordinates)

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
                              _plate_coordinates[0] + _plate_coordinates[2],
                              _plate_coordinates[1] + _plate_coordinates[3])]

    return _plate_coordinates


if __name__ == "__main__":
    for text_file in os.listdir(PLATES_PATH):
        text_file_path = os.path.join(PLATES_PATH, text_file)
        image_path = os.path.join(IMAGES_PATH, text_file.replace('txt', 'png'))
        character_coordinates = get_character_coordinates(text_file_path)
        plate_coordinates = get_plate_coordinates(text_file_path)
        new_text_path = os.path.join('/home/vitali/Projects/dataset/character_over_plates/texts', text_file)
        image = cv2.imread(image_path)
        cropped_image = image[plate_coordinates[0][1]: plate_coordinates[1][1],
                        plate_coordinates[0][0]:plate_coordinates[1][0]]

        with open(new_text_path, 'w') as _output_file:
            _output_file.write(str(len(character_coordinates)) + '\n')
            for coordinate in character_coordinates:
                new_coordinates = [(coordinate[0][0] - plate_coordinates[0][0],
                                    coordinate[0][1] - plate_coordinates[0][1]),
                                   (-plate_coordinates[0][0] + coordinate[1][0],
                                    -plate_coordinates[0][1] + coordinate[1][1])]


                _string = f'{new_coordinates[0][0]} {new_coordinates[0][1]} {new_coordinates[1][0]} {new_coordinates[1][1]}\n'
                _output_file.write(_string)

        new_image_path = os.path.join('/home/vitali/Projects/dataset/character_over_plates/images', text_file.replace('txt', 'png'))
        cv2.imwrite(new_image_path, cropped_image)
        print(new_image_path)
        # break
