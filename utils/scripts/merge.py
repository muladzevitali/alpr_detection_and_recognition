import os


PATH_MANY = '/home/vitali/files/football_project/train_dataset/text_lado'
PATH_FEW = '/home/vitali/files/football_project/train_dataset/logo_lado'

FEW = os.listdir(PATH_FEW)
MANY = os.listdir(PATH_MANY)

for file in FEW:
    with open(os.path.join(PATH_FEW, file), 'a') as new_file:
        with open(os.path.join(PATH_MANY, file), 'r') as old_file:
            lines = old_file.readlines()
        new_file.writelines(lines)
