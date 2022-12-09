#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load resting state functional data and convert to surface
# import
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
import numpy as np
import timeit
import glob
import os
import pandas as pd
from nilearn import signal
from scipy import stats
import nibabel as nib
from os.path import exists


# In[4]:


# how many files in the folder
my_folder = '/scratch/groups/jyeatman/HBN_FC/vol2surf/'
my_files = glob.glob(my_folder +'sub-*rest_lh.gii')
len(my_files)


# In[15]:


# Run on a subset of data for debugging
subs = ['sub-NDARAC349YUC']
datadir = '/scratch/groups/jyeatman/HBN_FC/'
# Load fsaverage
fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
# Functional folder
hemi = 'left' # options are 'left','right'
task = 'movie'
if task == 'rest':
    func1_suffix = '/functional_to_standard/_scan_rest_run-1/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/bandpassed_demeaned_filtered_antswarp.nii.gz'
elif task == 'movie':
    func1_suffix = '/functional_to_standard/_scan_movieDM/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/movieDM_bandpassed_demeaned_filtered_antswarp.nii.gz'


# In[16]:


# Load subject list
subject_file = datadir + 'subs_preprocessed_restingstate_movieDM_meanFD05_SNR15_FD05_90_WIAT_FilteredAfterScrubbing_0.5_0.2.csv'
if task == 'rest':
    subject_file = datadir+ 'subs_preprocessed_onlyrest_meanFD05_SNR15_FD05_90_WIAT.csv'
elif task == 'movie':
    subject_file = datadir+ 'subs_preprocessed_onlymovie_meanFD05_SNR15_FD05_90_WIAT.csv'

sub_data = pd.read_csv(subject_file)
subs = sub_data['participant_id']

sub_count = len(subs)
print(sub_count)


# In[ ]:


# TBD - make into a function that gets index list and use sherlock slurm job array to feed subjects in
# Loop over subjects
for ii in range(len(subs)): # for ii in indexlist:
    sub_dir = (datadir + 'input/' + subs.iloc[ii])
    surface_file = datadir + 'vol2surf/'+subs.iloc[ii] + '_task-' + task + '_' + hemi[0] + 'h.gii'
    surf_mesh=fsaverage['white_'+ hemi]
        
    if os.path.exists(surface_file):
        print(surface_file + ' already exists, skipping...')
    else:
        func1_file = sub_dir + func1_suffix
        if task == 'rest':
            func2_file = str.replace(func1_file, 'run-1', 'run-2')
            
        if os.path.exists(func1_file):
            print('Loading data for participant ' + subs.iloc[ii])   
            run1 = image.load_img(func1_file)
            # Convert to surface
            print('Converting run1 to vol')
            run1 = surface.vol_to_surf(img=run1,surf_mesh=surf_mesh,radius = 3)
            
            # Only for resting state, load second run and concatenate
            if task == 'rest':
                run2 = image.load_img(func2_file)
                print('Converting run2 to vol')
                run2 = surface.vol_to_surf(img=run2,surf_mesh=surf_mesh,radius = 3)
                # Concatenate the 2 runs
                run_data = np.concatenate((run1, run2),axis=1)
            elif task == 'movie':
                run_data = run1

            # Save as gifti
            print('Saving as gii')
           # nib.freesurfer.io.write_morph_data(datadir + 'TEST.gii',run_data)
            img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(run_data)])
            nib.save(img, surface_file)
        else:
            print(func1_file + ' doesnt exist, skipping...')

