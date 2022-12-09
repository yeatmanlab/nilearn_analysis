#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Run Functional connectivity on HBN data
# import
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
from nilearn import input_data
from nilearn import signal
from nilearn import masking
import numpy as np
import pandas as pd
import glob
import os
import nibabel as nib
from os.path import exists
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
#matplotlib.use('agg') # supposed to avoid memory leak - add to .py version of code when not running as notebook


# In[2]:


# Load subject fmri data
# This will loop across subjects
datadir = '/scratch/groups/jyeatman/HBN_FC/'
inputdir = datadir + 'input/'
outputdir = datadir + 'volume/'
#subs = glob.glob(datadir + 'sub-*') # this gets the full path
subs = [os.path.basename(x) for x in glob.glob(inputdir + '/sub-*')]
print('Found ' + str(len(subs)) + ' subjects in inputdir')


# In[10]:


# Where to save outputs: 
statsdir = outputdir + 'statMaps/'
imagedir =  outputdir + 'images/'

# Paths to surface ROIs based on Gari's coordinates (Lerma-Usabiaga et al., PNAS 2018)
# roidir =  '/home/groups/jyeatman/ROI_Atlases/' # old path
roidir = datadir 
roi_names = ['VWFA1','VWFA2']
roi_coords = [(-39,-71,-8),(-41,-60,-8)] # Lerma-Usabiaga et al., PNAS, 2018
#roi_coords = [(-44,-69,-12),(-41,-49,-20)] # White et al., PNAS 2019
roi_radius = 7 # 7mm to replicate my CONN analysis
# These are the coordinates from Lerma-Usabiaga, ...
# VWFA1 --> average of real words > checkerboards/scrambled words/phase scrambled words
# VWFA2 --> real words > pseudowords

# Scan parameters:
t_r = 0.8
droptp = 5

# Which task
task = 'movie' # options are 'rest' or 'movie'

# Steps to perform:
overwrite = False
createFigs = False # Do we want to generate figures at all- hopefully setting this to false will make it run fasterFalse  
saveFigs = False # If figures are to be created, save png files or display in notebook
saveMaps = True # Save actual connectivity map as a curv file that can 
# be loaded to Freeview
saveGroup = True # Save group data

# Cleaning parameters:
fd_thresh = 0.5
fd_vol_thresh = 20 # include only scans with >80% usable volumes
doScrubbing = False 
scrub_shift = True # scrub volumes adjacent to high FD volume - i-1, i, i+1, i+2
# Iliana says this practice has been abandoned recently to save data - trying it here just to replicate CONN analysis
# probably better to scrub only what's necessary
# TBD - Jason says that scrubbing should be done with multiple vectors, one per volumes, not a single vector

# Stat parameters
z_thresh = 0.25
cluster_thresh = 100 # Try 20/30 to get clean maps

# Load fsaverage
fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
# Load LH surface to get its size
white_left = surface.load_surf_data(fsaverage['white_left'])

# Load subject list
# subject file for original N=90 sample with both resting state and movie data
subject_file = datadir + 'subs_preprocessed_restingstate_movieDM_meanFD05_SNR15_FD05_90_WIAT_FilteredAfterScrubbing_0.5_0.2.csv'
# subject file for N=231
subject_file = datadir + 'subs_preprocessed_onlyrest_meanFD05_SNR15_FD05_90_WIAT.csv'
subject_file = datadir + 'subs_preprocessed_onlymovie_meanFD05_SNR15_FD05_90_WIAT.csv'
sub_list = pd.read_csv(subject_file)
subs = sub_list['participant_id']

# Run on a subset of data for debugging
# subs = ['sub-NDARAC349YUC']
subs = subs[0:2]
  
sub_count = len(subs)
print('Running on ' + str(sub_count) + ' subjects')

# Initialize group map to be used for statistics projected to surface
group_map = np.zeros(shape = (len(subs),white_left[0].shape[0],len(roi_names)))
print('Initializing group map shape: ' + str(group_map.shape))

# Define file paths
if task == 'rest':
    func1_suffix = '/functional_to_standard/_scan_rest_run-1/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/bandpassed_demeaned_filtered_antswarp.nii.gz'
    fd1_suffix = '/frame_wise_displacement_power/_scan_rest_run-1/FD.1d'
elif task == 'movie':
    func1_suffix = '/functional_to_standard/_scan_movieDM/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/movieDM_bandpassed_demeaned_filtered_antswarp.nii.gz'
    fd1_suffix = '/frame_wise_displacement_power/_scan_movieDM/FD.1d'
print(subs)


# In[12]:


# use mask that I had already created, no need to generate it everytime
# FIXME- use mask based on 90 subjects even when running on a subsample for debugging
# currently mask name is hardcoded (180 --> 90 subjects * 2 resting state runs)
mask_fileName = outputdir + 'groupMask_N' + str(180) + '.nii.gz'
# get size of brain
brain_mask_nii = image.load_img(mask_fileName)
brain_mask = brain_mask_nii.get_fdata()
brain = brain_mask[brain_mask==1]
group_map_vol = np.zeros(shape = (len(subs),len(brain.flatten()),len(roi_names)))
print('Initializing group map shape: ' + str(group_map_vol.shape))


# In[11]:


# Run analysis on volume
# Loop over subjects and compute connectivity for that ROI
for ii in range(len(subs)):  
    sub_dir = (inputdir + subs.iloc[ii]) # maybe we wont use that? 

    func1_file = sub_dir + func1_suffix

    if os.path.exists(func1_file):
        print('Loading data for participant ' + subs.iloc[ii])   
        run1 = image.load_img(func1_file)
        print('Original Data Volume by Timepoints')
        print(run1.shape) 
        run1 = image.index_img(run1,slice(droptp,-1))

        # Only for resting state, load second run and concatenate
        if task == 'rest':
            func2_file = str.replace(func1_file, 'run-1', 'run-2')
            if os.path.exists(func2_file):
                run2 = image.load_img(func2_file)
                run2 = image.index_img(run2,slice(droptp,-1))
                # Concatenate the 2 runs
                run_data = image.concat_imgs([run1,run2])
            else:
                run_data = run1
                print('Cant find second scan for ' + subs.iloc[ii])  
        else:
            run_data = run1
            print('Movie has a single run ' + subs.iloc[ii])
        
        if doScrubbing:
            FD_1 = pd.read_csv(sub_dir + fd1_suffix, header=None)
            FD_1 = FD_1.iloc[droptp+1: , :] 
            if task == 'rest':
                fd2_suffix = str.replace(fd1_suffix, 'run-1', 'run-2')
                if os.path.exists(fd2_suffix):
                    FD_2 = pd.read_csv(sub_dir + fd2_suffix, header=None)
                    FD_2 = FD_2.iloc[droptp+1: , :]
                    # Concatenate the 2 runs
                    FD = pd.concat([FD_1,FD_2],ignore_index=True)
                else:
                    print('Cant find second FD for ' + subs.iloc[ii])  
                    FD = FD_1
            elif task == 'movie':
                FD = FD_1
      
        print('After dropping initial 6 Timepoints, Volume by Timepoints')
        print(run_data.shape)
        
    else:    
        print('Cant find functional scan for ' + subs.iloc[ii])
        continue

    # Scrubbing
    if doScrubbing:
    # This implements a relatively restrictive scrubbing, removing adjacent volumes
    # Try to do this in a separate function  
    # TBD - For LMB data for example fix include scrubbing as a regressor
       scrub = scrubbing(FD, fd_thresh=0.5, fd_percent_thresh=80, scrub_method=None)
    # get brain-wide voxel-wise time series

    # For HBN data that was analyzed with CPAC I am not denoising with detrend
    # After changing the correlation calculation to pearson instead of dot product we don't 
    # need to standardize either
    
    brain_masker = input_data.NiftiMasker(smoothing_fwhm=None,
    detrend=False, standardize=False,
    low_pass=None, high_pass=None, t_r=t_r,
    memory=None, memory_level=1, verbose=0,
    mask_img=mask_fileName)

    # extract mean time series and regress out confounds
    # confounds=None becase CPAC already took care of that
    brain_time_series = brain_masker.fit_transform(run_data,confounds=None)
    print("Brain time series shape: (%s, %s)" % brain_time_series.shape)

#         # Plot mean timeseries
#         plt.plot(brain_time_series[:, [10, 45, 100, 5000, 10000 ]])
#         plt.title('Time series from 5 random voxels')
#         plt.xlabel('Scan number')
#         plt.ylabel('Signal')
#         plt.tight_layout()

     # extract time series for each roi, 7mm sphere around coordinates
    for c in range(len(roi_coords)):

        seed_masker = input_data.NiftiSpheresMasker(
        [roi_coords[c]], radius=roi_radius,
        detrend=False, standardize=False,
        low_pass=None, high_pass=None, t_r=t_r,smoothing_fwhm=None,
        memory=None, memory_level=1, verbose=0)

        # extract mean time series 
        seed_time_series = seed_masker.fit_transform(run_data,confounds=None)

#             fig,ax = plt.subplots(1)
#             plt.plot(seed_time_series)
#             plt.title('Time series from ROI #' + roi_names[c])
#             plt.xlabel('Scan number')
#             plt.ylabel('Signal')
#             plt.tight_layout()

        # Calculate seed-to-voxel correlation - original from nilearn tutorial
#        seed_map = (np.dot(brain_time_series.T, seed_time_series) /
#                     seed_time_series.shape[0])

        # Loop through voxels and use pearson - then no need to standardize!
        seed_map = np.zeros(brain_time_series.shape[1])
        for i in range(brain_time_series.shape[1]): # this loops through the voxels
            seed_map[i] = stats.pearsonr(seed_time_series.squeeze(), brain_time_series[:,i])[0]
            
        print("Seed-to-voxel correlation shape: (%s)" %
        seed_map.shape)
        print("Seed-to-voxel correlation: min = %.3f; max = %.3f" % (
        np.nanmin(seed_map), np.nanmax(seed_map)))

        # Fisher z-transformation
        seed_map_fisherZ = np.arctanh(seed_map)
        print("Seed-to-voxel correlation Fisher-z transformed: min = %.3f; max = %.3f"
              % (np.nanmin(seed_map_fisherZ),np.nanmax(seed_map_fisherZ)))

        # transform the correlation array back to a Nifti image object
        seed_map_fisherZ_img = brain_masker.inverse_transform(seed_map_fisherZ.T)
        
        # Plot image as volume - I prefer to convert to surface for visualization
#             display = plotting.plot_stat_map(seed_map_fisherZ_img,
#                                  threshold=0.5, vmax=2,
#                                  cut_coords=roi_coords[c],
#                                  title=(subs.iloc[ii] + ' ' + roi_names[c]))
#             display.add_markers(marker_coords=[roi_coords[c]], marker_color='slategray',marker_size=120)
        
        # convert to surface for visualization
        Zmap_surface = surface.vol_to_surf(img=seed_map_fisherZ_img,
                                                  surf_mesh=fsaverage['white_left'],radius = 3)
        # thresholding by cluster is only available in Volume space
        thresholded_map = image.threshold_img(img=seed_map_fisherZ_img,
                                              threshold = z_thresh,cluster_threshold=cluster_thresh)
        # convert to surface for visualization
        thresholded_surface = surface.vol_to_surf(img=thresholded_map,
                                                  surf_mesh=fsaverage['white_left'],radius = 3)
        
        # Save as a gifti that could be loaded into freeview
        if saveMaps:
            print('Saving correlation maps sub '+ subs.iloc[ii])
            # create filenames
            base_filename = statsdir + subs.iloc[ii] + '_task-' + task + '_' + roi_names[c] + '_zMap'
            targetNifti = base_filename + '.nii.gz'
            targetSurface = base_filename  + '.curv'
            targetNiftiThresh = base_filename + 'z_' + str(z_thresh)+ '_cluster_' + str(cluster_thresh) + '.nii.gz'
            targetSurfaceThresh = str.replace(targetNifti,'.nii.gz','.curv')

           #save data in appropriate filename - dont overwrite
            if not os.path.exists(targetNifti):
                seed_map_fisherZ_img.to_filename(targetNifti)
                print('writing ' + targetNifti)
            if not os.path.exists(targetSurface):
                nib.freesurfer.io.write_morph_data(targetSurface,Zmap_surface)
            if not os.path.exists(targetNiftiThresh):
                thresholded_map.to_filename(targetNiftiThresh)
            if not os.path.exists(targetSurfaceThresh):
                nib.freesurfer.io.write_morph_data(targetSurfaceThresh,thresholded_surface)

        # Add the stat map to the group stat map
        group_map_vol[ii,:,c] = seed_map_fisherZ.squeeze() # not sure how to use group_map_vol - for it to become
        # a nifti again we need to transform it to a volume shape, and then it's confusing to stack brain volumes 
        # one on top of the other, too many dimensions

        # do the same in surface
        group_map[ii,:,c] = Zmap_surface.squeeze()

        # Plot the seed-based connectivity
        figTitle = subs.iloc[ii] + ' ' + roi_names[c]
        output_subfilename = imagedir + subs.iloc[ii] + '_task-' + task + '_' + roi_names[c]

        if saveFigs:
            output_filel = output_subfilename + '_lateral.png'
            output_filev = output_subfilename + '_ventral.png'
            print('Saving ' + output_filel)
        else: 
            output_filel = None
            output_filev = None

        # Plot single subject map 
        plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=thresholded_surface,
        hemi='left', threshold = z_thresh, vmax=1, view='ventral', colorbar=True,
        bg_map=fsaverage['curv_left'], title=figTitle, output_file = output_filev)
        print()
        plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=thresholded_surface,
        hemi='left', threshold = z_thresh, vmax=1, view='lateral', colorbar=True,
        bg_map=fsaverage['curv_left'], title=figTitle, output_file = output_filel)
        print()

if doScrubbing:
    target_csv = str.replace(subject_file,'.csv','_scrubbing.csv')
    print(target_csv)
    sub_list.to_csv(target_csv)


# In[18]:


subs.iloc[ii]


# In[7]:


# save map of mean connectivity across the entire sample for each seed ROI
for c in range(len(roi_coords)):
    print('Calculating mean connectivity for ' + roi_names[c])
    group_mean = np.mean(group_map[:,:,c], axis = 0)
#    group_mean_vol = np.mean(group_map_vol[:,:,c], axis = 0)

    output_groupfilename = statsdir + 'GroupMap_task-' + task + '_' + roi_names[c] + '_N' + str(sub_count) + '_lateral.png'
    #save also as numpy array
    np.save(output_groupfilename[:-12],group_map_vol)

    # Save map of group mean
    if saveFigs:
        output_file = output_groupfilename
    else:
        output_file = None
    
    plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=group_mean,
    hemi='left', threshold = .25, vmax=0.7, view='lateral', colorbar=True,
    bg_map=fsaverage['curv_left'],title='Group map N=' + str(sub_count), output_file = output_file)
    #  plt.close()

    if saveGroup:
        group_filename = statsdir + 'GroupMean_task-' + task + roi_names[c] + '_N' + str(sub_count) + '.curv'
        nib.freesurfer.io.write_morph_data(group_filename,group_mean)


# In[25]:


#display = plotting.plot_img(seed_map_fisherZ_img)
myvals = seed_map_fisherZ_img.get_fdata().flatten()
plt.hist(myvals)


# In[1]:


# Testing this as a separate function outside of the loop
def scrubbing(FD, fd_thresh=0.5, fd_percent_thresh=80, scrub_method=None):
    """ this function is bla bla
    Input variables:
    FD: pandas dataframe single vecor
    fd_thresh
    etc
    """
    scrub = np.zeros(shape = (len(FD),1))    
    if scrub_method is None:
        return(scrub)
    elif scrub_method == 'old':
        scrub = np.zeros(shape = (len(FD),1))
        scrub[FD > fd_thresh] = 1
        if scrub_shift:
            scrub_orig = np.copy(scrub)
            indices = np.where(FD > fd_thresh)
            for a in indices[0]:
                scrub[a] = 1
                scrub[np.maximum(a-1,0)] = 1
                scrub[np.minimum(a+1,len(scrub)-1)] = 1
                scrub[np.minimum(a+2,len(scrub)-1)] = 1

             #   plt.plot(scrub_orig)
            #plt.plot(scrub)

        percent_scrub = sum(scrub)/len(scrub) *100
        sub_list.loc[ii,'scrub_num'] = sum(scrub)
        sub_list.loc[ii,'scrub_per'] = percent_scrub
        print('number of volumes to scrub: ' + str(sum(scrub)))
        if percent_scrub > fd_vol_thresh:
            print(subs.iloc[ii] + 'had ' + str(percent_scrub) + '% volumes scrubbed.')
            sub_list.loc[ii,'scrub_exclude'] = True
        else:
            sub_list.loc[ii,'scrub_exclude'] = False

        return(scrub)


# # Tests to browse different mask strategies

# In[67]:


report = brain_masker.generate_report()
report


# In[69]:


brain_masker = input_data.NiftiMasker(smoothing_fwhm=6,
detrend=True, standardize=True,
low_pass=None, high_pass=None, t_r=t_r,
memory=None, memory_level=1, verbose=0,
mask_strategy='whole-brain-template')
brain_time_series = brain_masker.fit_transform(run_data,confounds=None)
report = brain_masker.generate_report()
report


# In[73]:


brain_masker = input_data.NiftiMasker(smoothing_fwhm=6,
detrend=True, standardize=True,
low_pass=None, high_pass=None, t_r=t_r,
memory=None, memory_level=1, verbose=0,
mask_strategy='epi',mask_args=dict(upper_cutoff=.9, lower_cutoff=.8,
                                    opening=False))
brain_time_series = brain_masker.fit_transform(run_data,confounds=None)
report = brain_masker.generate_report()
report


# In[71]:


brain_masker = input_data.NiftiMasker(smoothing_fwhm=6,
detrend=True, standardize=True,
low_pass=None, high_pass=None, t_r=t_r,
memory=None, memory_level=1, verbose=0,
mask_strategy='background')
brain_time_series = brain_masker.fit_transform(run_data,confounds=None)
report = brain_masker.generate_report()
report


# In[72]:


brain_masker = input_data.NiftiMasker(smoothing_fwhm=6,
detrend=False, standardize=True,
low_pass=0.1, high_pass=0.01, t_r=t_r,
memory=None, memory_level=1, verbose=0,
mask_strategy='background')
brain_time_series = brain_masker.fit_transform(run_data,confounds=None)
report = brain_masker.generate_report()
report


# In[ ]:


plot_roi(masker.mask_img_, miyawaki_mean_img,
         title="Mask")


# In[175]:


display = plotting.plot_stat_map(seed_map_fisherZ_img,
                     threshold=0.3, vmax=2,
                     cut_coords=[-40,-60,10],
                     title=(subs.iloc[ii] + ' ' + roi_names[c] + ' orig'))


# In[176]:


display = plotting.plot_stat_map(thresholded_map,
                     threshold=0.3, vmax=2,
                     cut_coords=[-40,-60,10],
                     title=(subs.iloc[ii] + ' ' + roi_names[c] + ' thresh'))


# In[9]:


my_fig


# In[7]:


plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=correlation_map,
hemi='left', threshold = .3, vmax=2, view='lateral', colorbar=True,
bg_map=fsaverage['curv_left'], title='test', output_file = None)
print()


# In[17]:


# Plot single subject map - save lateral and ventral view in one figure
# Note that the png will have the two 'white' figures on a transparent background- 
# TBD find out how to make the background really white too
my_fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7,4),sharey=True,subplot_kw={'projection': '3d'})
plt.tight_layout()
plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=thresholded_surface,
hemi='left', threshold = .3, vmax=2, view='ventral', colorbar=False,axes=ax1,
bg_map=fsaverage['curv_left'], title=figTitle, output_file = None)

plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=thresholded_surface,
hemi='left', threshold = .3, vmax=2, view='lateral', colorbar=True,axes=ax2,
bg_map=fsaverage['curv_left'], title=figTitle, output_file = None)
print()
my_fig.savefig(output_filel) # make sure output_filel is not None!


# In[13]:


# nilearn function to plot directly from volume on the surface
plotting.plot_img_on_surf(stat_map=thresholded_map, surf_mesh='fsaverage',vmax=2,threshold=.3,
                            views=['lateral','ventral'])


# In[34]:


print(run1.shape)
run1 = image.index_img(run1,slice(droptp,-1))
print(run1.shape)


# In[39]:


plotting.plot_surf_stat_map(fsaverage['white_left'], stat_map=group_mean,
hemi='left', threshold = .25, vmax=0.7, view='lateral', colorbar=True,
bg_map=fsaverage['curv_left'],title='Group map N=' + str(sub_count), output_file = output_file)


# In[52]:


plotting.plot_surf_roi(fsaverage['white_left'], roi_map=cur_roi, hemi='left', view='lateral',
bg_map=fsaverage['curv_left'], bg_on_data=True, title='VWFA1',cmap='Reds')
print()


# # Create group brain mask

# In[6]:


# create a brain mask that fits all the subjects in the sample
# create a list of all functional scans
func_files = []
for ii in range(len(subs)):  
    sub_dir = (inputdir + subs.iloc[ii]) # maybe we wont use that? 
    func1_file = sub_dir + func1_suffix
    if os.path.exists(func1_file):
        func_files.append(func1_file)
    if task == 'rest':
        func2_file = str.replace(func1_file, 'run-1', 'run-2')
        func_files.append(func2_file)
print('Found ' + str(len(func_files)) + ' files')

# create a mask
group_mask = masking.compute_multi_background_mask(func_files)
# save
mask_fileName = outputdir + 'groupMask_N' + str(len(func_files)) + '.nii.gz'
group_mask.to_filename(mask_fileName)

