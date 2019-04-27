import os

PATH = '/home/vitali/Downloads/plates_bbox/Labels/check'


def get_coordinates(_row):
    coordinates = [int(each.strip()) for each in _row.split(' ') if each]
    x, y = coordinates[0], coordinates[1]
    new_x, new_y = x + coordinates[2], y + coordinates[3]
    return f'{x} {y} {new_x} {new_y}'


for text_file in os.listdir(PATH):
    text_file_path = os.path.join(PATH, text_file)
    with open(text_file_path, 'r') as _input_file:
        _info = _input_file.readlines()
    _info = [each.strip() for each in _info if each.strip()]
    with open(text_file_path, 'w') as _output_file:
        _output_file.write(_info[0] + '\n')
        coordinates_list = _info[1:]
        for _row in coordinates_list:
            print(len(coordinates_list))
            _row = get_coordinates(_row)
            _output_file.write(_row + '\n')


