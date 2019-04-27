import os


MAIN_PATH = '/home/vitali/files/football_project/unlabeled'
GIORGI = '/home/vitali/files/football_project/giorgi'
LADO = '/home/vitali/files/football_project/lado'
VITALI = '/home/vitali/files/football_project/vitali'

images = os.listdir(MAIN_PATH)
each_length = len(images) // 3
for i, image in enumerate(images):
    image_path = os.path.join(MAIN_PATH, image)
    if i < each_length:
        os.rename(image_path, os.path.join(GIORGI, image))
        continue
    elif i < 2 * each_length:
        os.rename(image_path, os.path.join(LADO, image))
        continue
    else:
        os.rename(image_path, os.path.join(VITALI, image))
