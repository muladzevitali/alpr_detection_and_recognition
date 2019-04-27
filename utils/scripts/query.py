import os

PATH = "/home/vitali/Projects/dataset"

IMAGES_PATH = os.path.join(PATH, 'images')
TEXTS_PATH = os.path.join(PATH, 'texts')

text_files = [file.split('.')[0] for file in os.listdir(TEXTS_PATH)]
images_files = [file.split('.')[0] for file in os.listdir(IMAGES_PATH)]

texts_set = set(text_files)
images_set = set(images_files)

print(texts_set.difference(images_set))