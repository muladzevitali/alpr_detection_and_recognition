import os

result_file_path = '/home/vitali/Downloads/plates_bbox/results'

base_path = '/home/vitali/Downloads/training'

for folder in os.listdir(base_path):
    data_path = os.path.join(base_path, folder)
    for item in os.listdir(data_path):
        if item.split('.')[-1] == 'txt':
            with open(os.path.join(data_path, item), 'r')as txt_file:
                lines = txt_file.readlines()
                number_of_plates = 0
                plates = []
                for line in lines:
                    if 'position_plate' in line:
                        number_of_plates += 1
                        coords = line.split(': ')[-1].split(' ')
                        coordinates = f'{coords[0]} {coords[1]} {int(coords[0])+int(coords[2])} ' \
                                      f'{int(coords[1])+int(coords[3])}'
                        plates.append(coordinates)

            with open(os.path.join(result_file_path, item), 'w') as result_txt:
                result_txt.write(str(number_of_plates)+'\n')
                for plate in plates:
                    result_txt.write(plate+'\n')
        else:
            os.rename(os.path.join(data_path, item), os.path.join(result_file_path, item))
