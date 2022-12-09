#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import matplotlib
import sys
#matplotlib.use('agg') # supposed to avoid memory leak - add to .py version of code when not running as notebook
#load_ext memory_profiler


# In[3]:


import platform
if platform.system() == "Darwin":
    projectdir = '/Volumes/GoogleDrive/My Drive/Projects/HBN/HBN_FC/'
    # Paths to surface ROIs from Emily
    roidir =  '/Users/mayayablonski/Documents/Data/FC/ROIs'
elif platform.system() == "Linux":
    projectdir = '/scratch/groups/jyeatman/HBN_FC/'
    # Paths to surface ROIs from Emily
    roidir =  '/home/groups/jyeatman/ROI_Atlases/visfAtlas/Emily/'


# In[33]:


# Load subject fmri data
# This will loop across subjects
# projectdir = '/scratch/groups/jyeatman/HBN_FC/'
inputdir = projectdir + 'input'
surfdir = projectdir + 'vol2surf/'
#subs = glob.glob(datadir + 'sub-*') # this gets the full path
subs = [os.path.basename(x) for x in glob.glob(inputdir + '/sub-*')]
print('Found ' + str(len(subs)) + ' subjects in inputdir')
subs = [os.path.basename(x) for x in glob.glob(surfdir + '/sub-*')]
print('Found ' + str(len(subs)) + ' subjects in surfdir')


# In[34]:


# Where to save outputs: 
surfacedir = projectdir + 'surface/statMaps/Faces/'
imagedir =  projectdir + 'surface/images/'

if not os.path.exists(surfacedir):
    os.makedirs(surfacedir)

if not os.path.exists(imagedir):
    os.makedirs(imagedir)
    
# Paths to surface ROIs from Kalanit's group - Rosenke 2021
#roidir =  '/home/groups/jyeatman/ROI_Atlases/visfAtlas/FreeSurfer/'
#roi_names = ['MPM_lh_OTS.label','MPM_lh_pOTS.label']

# Paths to surface ROIs based on Gari's coordinates (Lerma-Usabiaga PNAS 2018, converted vol2surf)
#roidir =  '/home/groups/jyeatman/ROI_Atlases/'
#roi_names = ['VWFA1.label.gii','VWFA2.label.gii']

#roi_names = ['lh_pOTS_chars.label','lh_mOTS_chars.label','MPM_lh_IOS.label']
roi_names = ['lh_pFus_faces.label','lh_mFus_faces.label']
# the IOS ROI is a character selective ROI from Rosenke 2021, looks like OWFA

# Which task
task = 'rest' # options are 'rest' or 'movie'
hemi = 'right' # right or left
seed_hemi = 'left'
# which correlation to save
corr_type = 'fisherz' # options are 'rval', 'fisherz'

overwrite = False
createFigs = False # create connectivity maps per subject
saveFigs = False   # Save png files of connectivity maps
saveMaps = False   # Save actual connectivity map as a curv file that can 
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

# parse command-line arguments; to use on Sherlock
print(sys.argv[1])
print(sys.argv[2])

start = int(sys.argv[1])
end = int(sys.argv[2])
subs = subs[start:end]

#subs = subs[10:150]
sub_count = len(subs)
print('Running on ' + str(sub_count) + ' subjects')


# In[35]:


subs


# In[36]:


# USE ATLAS ROIs?
# save these files in a separate folder
atlas = False
if atlas == True:
    roi_names = [b'S_front_inf',b'S_intrapariet_and_P_trans']
    roi_names= [b'G_pariet_inf-Supramar']
    roidir = 'Atlas'
    surfacedir = projectdir + 'surface/Atlas/statMaps/'
    imagedir =  projectdir + 'surface/Atlas/images/'
    if not os.path.exists(surfacedir):
        os.makedirs(surfacedir)
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    parcellation = destrieux_atlas['map_' + hemi]
    labels = destrieux_atlas['labels']


# When the seed ROI is not in the same hemisphere that we want to analyse, we actually need to load two surfaces:
# one to calculate the seed_timeseries of the source ROI, and another to extract all timeseries of all 
# vertices in the *other* hemisphere

for rr in range(len(roi_names)):
    # Allocate empty array for group data - N subjects * n vertices
    # We create a blank group_map here to avoid data bleeding from ROI to ROI in case there are missing values 
    if hemi == 'left':
        group_map = np.zeros(shape = (len(subs),white_left[0].shape[0]))
    elif hemi == 'right':
        group_map = np.zeros(shape = (len(subs),white_right[0].shape[0]))

    # Load ROI
    if atlas == False:
        cur_roi = surface.load_surf_data(roidir + roi_names[rr])
        cur_roi = cur_roi.astype(int) 
        print('Analyzing ROI ' + roidir + roi_names[rr])
    else:
        cur_roi = np.where(parcellation == labels.index(roi_names[rr]))[0]
        cur_roi = cur_roi.astype(int)
        # roi_names for the Atlas are bytes, not strings
        # (begin with 'b' --> need to be decoded inco UTF8)
        roi_names[rr] = roi_names[rr].decode()
        print('Analyzing ROI ' + roidir + ' ' + roi_names[rr])
        
    # Loop over subjects and compute connectivity for that ROI
    for ii in range(len(subs)):  
        output_subfilename = subs.iloc[ii] + '_task-' + task + '_' + roi_names[rr] + '_' + corr_type + '_' + hemi[0] + 'h.npy'
        if os.path.exists(surfacedir + output_subfilename):
            print(output_subfilename +' exists, \nskipping sub # ' + str(ii))
            stat_map = np.load(surfacedir + output_subfilename)
            group_map[ii,:] = stat_map 
            
        else:
            # Load data for full hemisphere
            func_file = subs.iloc[ii] + '_task-' + task + '_'+ hemi[0] + 'h.gii'
            if not exists(surfdir + func_file):
                print('Cant find ' + func_file)
                continue
            else:
                print('Loading sub # ' + str(ii) + ' ' + func_file)
                run_data = surface.load_surf_data(surfdir + func_file)
                # saving as gii and loading the surface seems to transpose the data
                # we want our data to be vertices * timepoints
                if run_data.shape[0] < run_data.shape[1]:
                    run_data = np.transpose(run_data)
                print('Original Data Vertices by Timepoints')
                print(run_data.shape) 
                # Drop the same volumes from the second run as well - only for resting state
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

            # Load data for seed ROI if different from target hemisphere    
            if seed_hemi == hemi:
                run_data_seed = run_data.copy()
            else:
                func_file_seed = subs.iloc[ii] + '_task-' + task + '_'+ seed_hemi[0] + 'h.gii'
                print('Loading sub # ' + str(ii) + ' ' + func_file_seed)
                run_data_seed = surface.load_surf_data(surfdir + func_file_seed)
                # saving as gii and loading the surface seems to transpose the data
                # we want our data to be vertices * timepoints
                if run_data_seed.shape[0] < run_data_seed.shape[1]:
                    run_data_seed = np.transpose(run_data_seed)
                print('Original Data Vertices by Timepoints')
                print(run_data_seed.shape) 
                # we want to drop the same volumes from the second run as well - only for resting state
                if task =='rest':
                    droptp2 = [x + int(run_data_seed.shape[1]/2) for x in droptp]
                    dropall = droptp + droptp2
                    print('dropping volumes: ' + str(dropall))

                elif task == 'movie':
                    dropall = droptp
                    print('dropping volumes: ' + str(dropall))

                run_data_seed = np.delete(run_data_seed,dropall,axis=1)
                print('After dropping initial 6 Timepoints, Vertices by Timepoints')
                print(run_data_seed.shape) 
                
                
            # Compute the mean time series for the ROI
            seed_timeseries = np.nanmean(run_data_seed[cur_roi], axis=0)
            # To plot mean timeseries:
            fig, ax = plt.subplots(figsize =(4, 3))
            ax.plot(seed_timeseries)
            ax.set_title('Seed timeseries ' + subs.iloc[ii] + ' ' + roi_names[rr])
            ax.set_xlabel('Volume number')
            ax.set_ylabel('Normalized signal')
            print()

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
                targetFile = surfacedir + subs.iloc[ii] + '_task-' + task + '_' + roi_names[rr] + '_' + corr_type + '_'+hemi[0]+'h.curv'
                if not os.path.exists(targetFile):
                    nib.freesurfer.io.write_morph_data(targetFile,stat_map)

            # Add the stat map to the group stat map
            group_map[ii,:] = stat_map 

            # Save individual correlation maps - this way we can use a csv with 
            # a list of subjects and create a group npy from selected subjects
            np.save(surfacedir + output_subfilename,stat_map)

            # Plot the seed-based connectivity
            figTitle = subs.iloc[ii] + ' ' + roi_names[rr]

            if createFigs:
                if saveFigs:
                    output_filel = imagedir+os.path.splitext(output_subfilename)[0] + '_lateral.png'
                    output_filev = imagedir+os.path.splitext(output_subfilename)[0] + '_ventral.png'
                    print('Saving ' + output_filel)
                else: 
                    output_filel = None
                    output_filev = None

                plotting.plot_surf_stat_map(fsaverage['white_' + hemi], stat_map=stat_map,
                hemi=hemi, threshold = .3, vmax=0.7, view='lateral', colorbar=True,
                bg_map=fsaverage['curv_'+ hemi], title=figTitle, output_file = output_filel)
                print()
                #plt.close()

                plotting.plot_surf_stat_map(fsaverage['white_' + hemi], stat_map=stat_map,
                hemi=hemi, threshold = .3, vmax=0.7, view='ventral', colorbar=True,
                bg_map=fsaverage['curv_' + hemi], title=figTitle, output_file = output_filev)
                print()
                #plt.close()
            
    # outside the subject loop, save map of mean connectivity across the entire sample
    print('Calculating mean connectivity for ' + roi_names[rr])
    group_mean = np.mean(group_map, axis = 0)

    # Save map of group mean
    output_groupfilename = surfacedir + 'GroupMap_task-' + task + '_' + roi_names[rr] + '_N' + str(len(subs)) + '_' + corr_type + '_' + hemi[0]+'h'
    if saveFigs:
        output_file = output_groupfilename+ '_lateral.png'
        output_file_v = str.replace(output_file,'lateral','ventral')
    else:
        output_file = None
        output_file_v = None

    plotting.plot_surf_stat_map(fsaverage['white_' + hemi], stat_map=group_mean,
    hemi=hemi, threshold = .25, vmax=0.7, view='lateral', colorbar=True,
    bg_map=fsaverage['curv_' + hemi],title='Group map N=' + str(len(subs)), output_file = output_file)
    
    plotting.plot_surf_stat_map(fsaverage['white_' + hemi], stat_map=group_mean,
    hemi=hemi, threshold = .25, vmax=0.7, view='ventral', colorbar=True,
    bg_map=fsaverage['curv_' + hemi],title='Group map N=' + str(len(subs)), output_file = output_file_v)
    
    if saveGroup:
        nib.freesurfer.io.write_morph_data(output_groupfilename+ '.curv',group_mean)        
        #save also as numpy array
        np.save(output_groupfilename,group_map)

