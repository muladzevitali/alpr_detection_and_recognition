import os
from os import walk, getcwd
from PIL import Image

classes = ['ALPR']
cls = "ALPR"


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


""" Configure Paths"""
mypath = "/home/vitali/Projects/dataset/plates_over_cars/Labels/001/"
outpath = "/home/vitali/Projects/dataset/plates_over_cars/yolo_plates_over_cars/"
IMAGE_PATH = '/home/vitali/Projects/dataset/plates_over_cars/Images/001/'

if cls not in classes:
    exit(0)
cls_id = classes.index(cls)

wd = getcwd()
list_file = open(f'{wd}/{cls}_list.txt', 'w')

""" Get input text file list """
txt_name_list = []
for dir_path, dir_names, file_names in walk(mypath):
    txt_name_list.extend(file_names)
    break
print(txt_name_list)

""" Process """
for txt_name in txt_name_list:
    """ Open input text files """
    txt_path = mypath + txt_name
    print("Input:" + txt_path)
    txt_file = open(txt_path, "r")
    lines = txt_file.read().split('\n')  # for ubuntu, use "\r\n" instead of "\n"

    """ Open output text files """
    txt_out_path = outpath + txt_name
    print("Output:" + txt_out_path)
    txt_outfile = open(txt_out_path, "a")

    """ Convert the data to YOLO format """
    ct = 0
    for line in lines:
        elems = line.split(' ')
        if len(elems) >= 4:
            ct = ct + 1

            xmin = elems[0]
            xmax = elems[2]
            ymin = elems[1]
            ymax = elems[3]
            image_path = os.path.join(IMAGE_PATH,
                                      f'{os.path.splitext(txt_name)[0]}.png')
            im = Image.open(image_path)
            w = int(im.size[0])
            h = int(im.size[1])

            b = (float(xmin), float(xmax), float(ymin), float(ymax))
            bb = convert((w, h), b)
            txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    """ Save those images with bb into list"""
    if ct != 0:
        list_file.write(f'{wd}/images/{cls}/{os.path.splitext(txt_name)[0]}.png\n')

list_file.close()
