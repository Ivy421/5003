import gaitevents
import pickle

pkl = 'E:/NUS/5003/gaitanalysisvideo-main/AR Gait_P_H016_Free walk_25-11-2022_15-22-38_noAR.pkl'
leg_length = 0.8
    
gaitdata = gaitevents.load_object(pkl)

gaitdata.patientinfo = {
    'leg_length_left': leg_length,
    'leg_length_right': leg_length}

# print("Processing CBTA for "+pkl)
gaitevents.cbta(gaitdata)



