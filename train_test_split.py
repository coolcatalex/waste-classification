import glob
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

def get_filenames_in_folder_and_subfolders(folder_path):
    pattern = os.path.join(folder_path, '**', '*')
    filenames = [file for file in glob.glob(pattern, recursive=True) if os.path.isfile(file)]
    return filenames

# Example usage
folder_path = '/home/pankaj/IC_models/trashNet/dataset'
all_filenames = get_filenames_in_folder_and_subfolders(folder_path)
print(len(all_filenames))
# df_file=pd.DataFrame(all_filenames, columns=['filenames'])
train_images, val_images = train_test_split(all_filenames, test_size=0.2, random_state=42)
print(len(train_images),len(val_images))
# df_file.to_csv('all_filenames.csv',index=False)

for img in train_images:
    # src = os.path.join(cls_dir, img)
    # dst = os.path.join(train_dir, cls, img)
    # shutil.move(src, dst)
    print(img)
    classname=img.split('/')[-2]
    filename=img.split('/')[-1]
    print(classname, filename)

    if not os.path.exists(f'/home/pankaj/IC_models/trashNet/dataset/train_resized/{classname}'):
        os.makedirs(f'/home/pankaj/IC_models/trashNet/dataset/train_resized/{classname}')

    shutil.copy(img, f'/home/pankaj/IC_models/trashNet/dataset/train_resized/{classname}/{filename}')
    print(img)
    print(f'/home/pankaj/IC_models/trashNet/dataset/train_resized/{classname}/{filename}')

img=''
for img in val_images:
    # src = os.path.join(cls_dir, img)
    # dst = os.path.join(train_dir, cls, img)
    # shutil.move(src, dst)
    print(img)
    classname=img.split('/')[-2]
    filename=img.split('/')[-1]
    print(classname, filename)

    if not os.path.exists(f'/home/pankaj/IC_models/trashNet/dataset/test_resized/{classname}'):
        os.makedirs(f'/home/pankaj/IC_models/trashNet/dataset/test_resized/{classname}')

    shutil.copy(img, f'/home/pankaj/IC_models/trashNet/dataset/test_resized/{classname}/{filename}')
    print(img)
    print(f'/home/pankaj/IC_models/trashNet/dataset/test_resized/{classname}/{filename}')

    # break