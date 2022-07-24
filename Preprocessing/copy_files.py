# this file is used to copy only the 818 mri images 
import pandas as pd
import shutil
data = pd.read_csv('C:/Users/yacin/Desktop/PFE/PFE Code/csv/init_test.csv')
SOURCE = 'E:/Yacine/Dataset/ADNI/'
TARGET = 'C:/Users/yacin/Desktop/PFE/ADNI/'

for file in data['Image Data ID'].tolist():
    shutil.copy2(SOURCE+file+'.nii', TARGET)