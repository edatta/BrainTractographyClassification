#!/usr/bin/env python

# streamline_labels.py is used to set up a data set of brain tractography data with ground truth labels.

import nibabel as nib
import numpy as np
import glob

def streamline_labels(set_file, label_status):

    patient_files = sorted(glob.glob(set_file+'/*/*'))
    #print(patient_files)
    all_patients_streamlines = []      
        
    for patient_file in patient_files:

        all_trk_files = sorted(glob.glob(patient_file+'/*Ex_Capsule_L*'))
        all_trk_file = all_trk_files[0]
        IFOF_trk_files = sorted(glob.glob(patient_file+'/*good_IFOF_L*'))
        IFOF_trk_file = IFOF_trk_files[0]
        UNC_trk_files = sorted(glob.glob(patient_file+'/*good_Uncinate_L*'))
        UNC_trk_file = UNC_trk_files[0]

        all_trk, all_hdr = nib.trackvis.read(all_trk_file)
        all_streamlines = [item[0] for item in all_trk]
        IFOF_trk, IFOF_hdr = nib.trackvis.read(IFOF_trk_file)
        IFOF_streamlines = [item[0] for item in IFOF_trk]
        UNC_trk, UNC_hdr = nib.trackvis.read(UNC_trk_file)
        UNC_streamlines = [item[0] for item in UNC_trk]

        all_patients_streamlines = all_patients_streamlines + all_streamlines

        if(label_status == False):

            labels = np.zeros(len(all_streamlines))

            if(len(IFOF_streamlines) != 0):  
                for IFOF_streamline in IFOF_streamlines:
                    for j in xrange(len(all_streamlines)):
                        if(np.array_equal(IFOF_streamline,all_streamlines[j])):
                            labels[j] = 1
            
            if(len(UNC_streamlines) != 0):
                for UNC_streamline in UNC_streamlines:
                    for j in xrange(len(all_streamlines)):
                        if(np.array_equal(UNC_streamline,all_streamlines[j])):
                            labels[j] = 2

            print(patient_file)
            #print('Total Streamlines')
            #print(len(all_streamlines))

            if(len(IFOF_streamlines) != (np.sum(labels == 1))):
                print('IFOF Streamlines')
                print(len(IFOF_streamlines))
                print('IFOF Found in Ex_Capsule_L')
                print(np.sum(labels == 1))
            
            if(len(UNC_streamlines) != (np.sum(labels == 2))):
                print('Uncincate Streamlines')
                print(len(UNC_streamlines))
                print('Uncinate Found in Ex_Capsule_L')
                print(np.sum(labels == 2)) 

            if (patient_file == patient_files[0]):
                all_labels = labels
            else:
                all_labels = np.hstack((all_labels, labels))
         
    if(label_status == True):
        all_labels = np.loadtxt(set_file+'/labels.txt') 
    else:
        np.savetxt(set_file+'/labels.txt', all_labels)
    
    return (all_patients_streamlines, all_labels)
