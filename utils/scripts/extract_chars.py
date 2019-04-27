import os

result_file_path = '/home/vitali/Downloads/plates_bbox/results/'

base_path = '/home/vitali/Downloads/training/'
files = os.listdir(base_path)
for file in files:
    data_path = os.path.join(base_path, file)
    data = os.listdir(data_path)
    for item in data:
        if item.endswith('txt'):
            try:
                with open(os.path.join(data_path, item), 'r')as txt_file:
                    lines = txt_file.readlines()
                    number_of_plates = 0
                    plates = []
                    for line in lines:
                        if 'char' in line:
                            number_of_plates += 1
                            coords = line.split(': ')[-1].split()
                            coordinates = f'{coords[0]} {coords[1]} {int(coords[0])+int(coords[2])} ' \
                                          f'{int(coords[1])+int(coords[3])}'
                            plates.append(coordinates)

                with open(os.path.join(result_file_path, item), 'w') as result_txt:
                    result_txt.write(str(number_of_plates) + '\n')
                    for plate in plates:
                        result_txt.write(plate + '\n')
                os.rename(os.path.join(data_path, item.replace('txt', 'png')),
                          os.path.join(result_file_path, item.replace('txt', 'png')))
            except:
                continue
