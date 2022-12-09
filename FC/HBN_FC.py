#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Run Functional connectivity on HBN data
# import
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
import numpy as np
import pandas as pd
import glob
import os
from nilearn import signal
from scipy import stats
import nibabel as nib
from os.path import exists
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use('agg') # supposed to avoid memory leak - add to .py version of code when not running as notebook
#get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[7]:


# Load subject fmri data
# This will loop across subjects
projectdir = '/scratch/groups/jyeatman/HBN_FC/'
inputdir = projectdir + 'input'
#subs = glob.glob(datadir + 'sub-*') # this gets the full path
subs = [os.path.basename(x) for x in glob.glob(inputdir + '/sub-*')]
print('Found ' + str(len(subs)) + ' subjects in inputdir')


# In[11]:


# Where to save outputs: 
surfacedir = projectdir + 'surface/statMaps/'
imagedir =  projectdir + 'surface/images/'

if not os.path.exists(surfacedir):
    os.makedirs(surfacedir)

if not os.path.exists(imagedir):
    os.makedirs(imagedir)
    
# Paths to surface ROIs from Kalanit's group - Rosenke 2021
roidir =  '/home/groups/jyeatman/ROI_Atlases/visfAtlas/FreeSurfer/'
roi_names = ['MPM_lh_OTS.label','MPM_lh_pOTS.label']

# Paths to surface ROIs based on Gari's coordinates (Lerma-Usabiaga PNAS 2018, converted vol2surf)
roidir =  '/home/groups/jyeatman/ROI_Atlases/'
roi_names = ['VWFA1.label.gii','VWFA2.label.gii']

# Paths to surface ROIs from Emily
roidir =  '/home/groups/jyeatman/ROI_Atlases/visfAtlas/Emily/'
roi_names = ['lh_pOTS_chars.label','lh_mOTS_chars.label','MPM_lh_IOS.label']
# the IOS ROI is a character selective ROI from Rosenke 2021, looks like OWFA

# Which task
task = 'rest' # options are 'rest' or 'movie'
hemi = 'left'
# which correlation to save
corr_type = 'fisherz' # options are 'rval', 'fisherz'

overwrite = False
createFigs = False # create connectiovity maps per subject
saveFigs = False # Save png files of connectivity maps
saveMaps = False # Save actual connectivity map as a curv file that can 
# be loaded to Freeview
saveGroup = True

# Run on a subset of data for debugging
subs = ['sub-NDARAC349YUC']

droptp = [0,1,2,3,4,5]
fd_thresh = 0.5
fd_vol_thresh = 90 # include only scans with >90% usable volumes

# Load fsaverage
fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
# Load LH surface to get its size
white_left = surface.load_surf_data(fsaverage['white_left'])
# Load RH surface to get its size
white_right = surface.load_surf_data(fsaverage['white_right'])

# Load subject list
subject_file = projectdir+ 'subs_preprocessed_restingstate_movieDM_meanFD05_SNR15_FD05_90_WIAT_FilteredAfterScrubbing_0.5_0.2.csv'
if task == 'rest':
    subject_file = projectdir+ 'subs_preprocessed_onlyrest_meanFD05_SNR15_FD05_90_WIAT.csv'
elif task == 'movie':
    subject_file = projectdir+ 'subs_preprocessed_onlymovie_meanFD05_SNR15_FD05_90_WIAT.csv'
sub_list = pd.read_csv(subject_file)
subs = sub_list['participant_id']
#subs = subs[0:3]

# Allocate empty array for group data - N subjects * n vertices
group_map = np.zeros(shape = (len(subs),white_left[0].shape[0]))
sub_count = len(subs)
print('Running on ' + str(sub_count) + ' subjects')


# In[12]:


# try to turn into separate functions modularize code and test each part


# In[16]:


# Try loading a gii surface file that was previously saved
for rr in range(len(roi_names)):
    # Load first ROI
    cur_roi = surface.load_surf_data(roidir + roi_names[rr])
    cur_roi = cur_roi.astype(int) # Does this line do something unexpected?!
#     cur_roi = surface.load_surf_data(roidir + roi_names[rr] + '.label') # backup original line
    print('Analyzing ROI ' + roidir + roi_names[rr])
    
    # Loop over subjects and compute connectivity for that ROI
    for ii in range(len(subs)):  
        sub_dir = (inputdir + subs.iloc[ii]) # maybe we won't use sub_dir for anything? 
        func_file = subs.iloc[ii] + '_task-' + task + '.gii'
        if not exists(projectdir + func_file):
            print('Cant find ' + func_file)
            continue
        
        print('Loading sub # ' + str(ii) + ' ' + func_file)
        run_data = surface.load_surf_data(projectdir + func_file)
        # saving as gii and loading the surface seems to transpose the data
        # we want our data to be vertices * timepoints
        if run_data.shape[0] < run_data.shape[1]:
            run_data = np.transpose(run_data)
        print('Original Data Vertices by Timepoints')
        print(run_data.shape) 
        # we want to drop the same volumes from the second run as well - only for resting state
        if task =='rest':
            droptp2 = [x + int(run_data.shape[1]/2) for x in droptp]
            dropall = droptp + droptp2
            print('dropping volumes: ' + str(dropall))
 
        elif task == 'movie':
            dropall = droptp
            print('dropping volumes: ' + str(dropall))

        run_data = np.delete(run_data,dropall,axis=1)
        print('After dropping initial 6 Timepoints, Vertices by Timepoints')
        print(run_data.shape) 
        # TBD - implement scrubbing

        # Compute the mean time series for the ROI
        seed_timeseries = np.nanmean(run_data[cur_roi], axis=0)
#        # To plot mean timeseries:
        # fig, ax = plt.subplots(figsize =(4, 3))
        # ax.plot(seed_timeseries)
        # ax.set_title('Seed timeseries ' + subs.iloc[ii] + ' ' + roi_names[rr])
        # ax.set_xlabel('Volume number')
        # ax.set_ylabel('Normalized signal')
        # print()
        
        # Compute correlations between the seed timeseries and each vertex
        stat_map = np.zeros(run_data.shape[0])
        for i in range(run_data.shape[0]): # this loops through the vertices
            stat_map[i] = stats.pearsonr(seed_timeseries, run_data[i])[0]

        print('computing stat_map ' + subs.iloc[ii])
        if corr_type == 'fisherz':
        # Fisher transform the map
            stat_map = np.arctanh(stat_map)

        # Save as a gifti that could be loaded into freeview
        if saveMaps:
            targetFile = surfacedir + subs.iloc[ii] + '_task-' + task + '_' + roi_names[rr] + corr_type + '.curv'
            if not os.path.exists(targetFile):
                nib.freesurfer.io.write_morph_data(targetFile,stat_map)

        # Add the stat map to the group stat map
        group_map[ii,:] = stat_map 
        
        # Plot the seed-based connectivity
        figTitle = subs.iloc[ii] + ' ' + roi_names[rr]
        output_subfilename = imagedir + subs.iloc[ii] + '_task-' + task + '_' + roi_names[rr] + corr_type

        if createFigs:
            if saveFigs:
                output_filel = output_subfilename + '_lateral.png'
                output_filev = output_subfilename + '_ventral.png'
                print('Saving ' + output_filel)
            else: 
                output_filel = None
                output_filev = None

            plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=stat_map,
            hemi='left', threshold = .3, vmax=0.7, view='lateral', colorbar=True,
            bg_map=fsaverage['curv_left'], title=figTitle, output_file = output_filel)
            print()
            #plt.close()

            plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=stat_map,
            hemi='left', threshold = .3, vmax=0.7, view='ventral', colorbar=True,
            bg_map=fsaverage['curv_left'], title=figTitle, output_file = output_filev)
            print()
            #plt.close()
        
#         # save intermediate map of mean connectivity as we add more subjects

        print('Calculating mean connectivity for ' + roi_names[rr])
        if ii > 0:
            mid_group_mean = np.mean(group_map[0:ii,:], axis = 0)

            mid_groupfilename = surfacedir + 'mid_GroupMap_task-' + task + '_' + roi_names[rr] + '_N' + str(ii) + '_lateral.png'
            #save also as numpy array
            np.save(mid_groupfilename[:-12],group_map)
            
    # outside the subject loop, save map of mean connectivity across the entire sample
    print('Calculating mean connectivity for ' + roi_names[rr])
    group_mean = np.mean(group_map, axis = 0)

    # Save map of group mean
    if saveFigs:
        output_file = output_groupfilename+ '_lateral.png'
    else:
        output_file = None

    plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=group_mean,
    hemi='left', threshold = .25, vmax=0.7, view='lateral', colorbar=True,
    bg_map=fsaverage['curv_left'],title='Group map N=' + str(sub_count), output_file = output_file)
  #  plt.close()
    
    if saveGroup:
        output_groupfilename = surfacedir + 'GroupMap_task-' + task + '_' + roi_names[rr] + '_N' + str(sub_count) + '_' + corr_type
        nib.freesurfer.io.write_morph_data(output_groupfilename+ '.curv',group_mean)        
        #save also as numpy array
        np.save(output_groupfilename,group_map)

