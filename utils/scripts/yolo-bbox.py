import os
from os import getcwd
from PIL import Image
import shutil


classes = ["Text", "Logo"]
current_class = "Text"


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]

    xmax = (2 * box[0] + box[2]) / 2
    xmin = (2 * box[0] - box[2]) / 2
    ymax = (2 * box[1] + box[3]) / 2
    ymin = (2 * box[1] - box[3]) / 2

    xmax /= dw
    xmin /= dw
    ymax /= dh
    ymin /= dh
    return int(xmin), int(ymin), int(xmax), int(ymax)


""" Configure Paths"""
YOLO_labels = "/home/vitali/files/football_project/train_dataset/yolov3/text-logos"
BBox_labels = "/home/vitali/files/BBox-Label-Tool/Labels/001"
BBox_images = '/home/vitali/files/BBox-Label-Tool/Images/001'


if current_class not in classes:
    exit()
cls_id = classes.index(current_class)

working_directory = getcwd()
list_file = open(f'{working_directory}/{current_class}_list.txt', 'w')

""" Get input text file list """
txt_name_list = []
for file in os.listdir(YOLO_labels):
    if file.endswith('.txt'):
        txt_name_list.append(file)


""" Process """
for txt_name in txt_name_list:
    """ Open input text files """
    txt_path = os.path.join(YOLO_labels, txt_name)
    txt_file = open(txt_path, "r")
    lines = txt_file.read().split('\n')
    image_path = os.path.join(YOLO_labels, txt_name.split('.')[0] + '.jpg')

    """ Open output text files """
    txt_out_path = os.path.join(BBox_labels, txt_name)
    txt_outfile = open(txt_out_path, "w")

    """ Convert the data to BBox format """
    boxes = 0
    output_coordinates = []
    for line in lines:
        coordinates = line.split(' ')
        if coordinates[0] == str(cls_id):
            boxes += 1
            x = float(coordinates[1])
            y = float(coordinates[2])
            w = float(coordinates[3])
            h = float(coordinates[4])
            image = Image.open(image_path)
            width = int(image.size[0])
            height = int(image.size[1])
            b = (x, y, w, h)
            bbox = convert((width, height), b)
            output_coordinates.append(bbox)

    txt_outfile.write(str(boxes) + '\n')
    for each in output_coordinates:
        txt_outfile.write(f'{each[0]} {each[1]} {each[2]} {each[3]}\n')

    shutil.copyfile(image_path, os.path.join(BBox_images, image_path.split('/')[-1]))

list_file.close()
