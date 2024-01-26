import gaitevents
import pickle
import os

leg_length = 0.8

folder_path = 'E:/NUS/5003/pickle_file'

def find_pkl_files(folder_path):
    pkl_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files
all_pkl_files = find_pkl_files(folder_path)
#print(all_pkl_files)

for pkl in all_pkl_files:
    print('processing:%s'%pkl)
    gaitdata = gaitevents.load_object(pkl)
    gaitdata.patientinfo = {
            'leg_length_left': leg_length,
            'leg_length_right': leg_length}
    gaitevents.cbta(gaitdata,filename = pkl)
