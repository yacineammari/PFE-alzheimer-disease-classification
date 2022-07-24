# this file take the path the ADNI dataset then change the rename every file to it image id 
import os

path = input('please input the path to the ADNI dataset: ')

for file in os.listdir(path):

    try:
        if file.lower().endswith(('.nii' )):
            old_name = file.replace('.nii','')
            new_name = old_name.split('_')[-1]+'.nii'
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
    except:
        pass