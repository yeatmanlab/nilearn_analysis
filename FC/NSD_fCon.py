#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import nilearn and numpy
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn import image
from nilearn import masking
import numpy as np
import pandas as pd
import glob
from nilearn import signal
from scipy import stats
import nibabel as nib
import glob
import os
sk = 3
import matplotlib.pyplot as plt


# In[2]:


# install nsdcode by cloning the repository and following the instructions:
# https://github.com/cvnlab/nsdcode/
from nsdcode.nsd_mapdata import NSDmapdata
from nsdcode.nsd_datalocation import nsd_datalocation
from nsdcode.nsd_output import nsd_write_fs
from nsdcode.utils import makeimagestack


# In[14]:


# List all paths
import platform
if platform.system() == "Darwin":
    basepath = '/Users/mayayablonski/Sherlock/groupScratch/'
elif platform.system() == "Linux":
    basepath = '/scratch/groups/jyeatman/'
    basepath = '/oak/stanford/groups/jyeatman/'
    
basedir = basepath + 'NSD/ppdata/'
fsdir = basepath + 'NSD/nsddata/freesurfer/'
analysisdir = os.path.dirname(os.path.dirname((basedir))) + '/analysis/'
outputdir = analysisdir + 'clean_18P_scrub025_butter/'
nsd_path = basepath + '/NSD/'
nsd = NSDmapdata(nsd_path)

print(basedir)


# In[15]:


# Set parameters
clobber = False

hemi='lh'
seed_hemi = 'lh'

droptp = [0,1,2,3,4,5]
subs = ['subj01','subj02','subj03','subj04','subj05','subj06','subj07','subj08']
roi_names = ['vwfa1','vwfa2']
roi_labels = [2,3]

# new labels
new_labels = True
if new_labels:
    roi_names = ['vwfa1','vwfa2','IFSwords','IPSwords']

roi_names = ['VWFA1','VWFA2','IFSwords','IPSwords']

faces = True
if faces:
    roi_names = ['FFA1_lh','FFA2_lh']
    
# Denoising
confound_list = ["WMe4","CSF","GSR",'trans_y','trans_x','trans_z', 'rot_x','rot_y','rot_z']
use_conf_diff = True
use_conf_quad = False
apply_scrubbing = True
myfilter = 'butterworth'
for reg in range(len(confound_list)):
    if use_conf_diff:
        new_names = [confound_list[reg]+'_derivative']
        confound_list.extend(new_names)
    if use_conf_quad:
        new_names = [confound_list[reg]+'_quadratic',confound_list[reg]+'_quadratic_derivative']
        confound_list.extend(new_names)

print(confound_list)

# map volume to surface space
sourcespace = 'func1pt8'
targetspace = hemi+'.layerB2'
seed_targetspace = seed_hemi+'.layerB2'
interpmethod = 'cubic'


# In[16]:


# Syntax to run on subset of data for debugging
# runs = runs[0:1]
# runs = ['/func1pt8mm/timeseries/timeseries_session21_run01.nii.gz', '/func1pt8mm/timeseries/timeseries_session21_run14.nii.gz']
# runs = glob.glob(basedir + subs[0] + '/func1pt8mm/timeseries/timeseries*nii.gz')
# runs.sort()
# print('Found ' + str(len(runs)) + ' runs for sub ' + subs[0])
# subs=subs[0:1]
subs


# In[28]:


for s,sub in enumerate(subs):
    fssub = fsdir + sub
    bolddir = basedir + sub + '/func1pt8mm/timeseries/'
    motiondir = basedir + sub + '/func1pt8mm/motion/'
    
    surf = surface.load_surf_mesh(fssub + '/surf/'+hemi+'.white')
    labels = surface.load_surf_data(fssub + '/label/'+hemi+'.floc-words.mgz')
    curv = surface.load_surf_data(fssub + '/surf/'+hemi+'.curv')
    
    runs = glob.glob(bolddir + '/timeseries*nii.gz')
    runs.sort()
    print('Found ' + str(len(runs)) + ' runs for sub ' + sub)
    # Changed zeros to NaNs for more accurate averaging
    roi_zmap = np.empty([surf.coordinates.shape[0], len(runs),len(roi_names)])
    roi_zmap[:] = np.NaN
    
    for r, run in enumerate(runs[0:2]):
        
        # load presaved confounds - WM and CSF regressors
        confound_file = str.replace(run,'timeseries','conf')
        confound_file = str.replace(confound_file,'.nii.gz','.tsv')
        
        if os.path.exists(confound_file):
            print('Loading confound regressors: '+ confound_file)
            confounds_all = pd.read_csv(confound_file,sep='\t',header=0)
        else:
            print('Cant locate: ' + confound_file)
            continue
        
        # TESTME - select only the desired confounds
        confounds = confounds_all.loc[:,confound_list]
        # combine with scrubbing regressors
        if apply_scrubbing:
            scrubbing_confounds = confounds_all.loc[:, confounds_all.columns.str.startswith('scrub_')]
            confounds=pd.merge(confounds,scrubbing_confounds,left_index=True,right_index=True)

        # drop initial timepoints
        confounds = confounds.iloc[len(droptp): , :]
                
        brain_bold_path = str.replace(analysisdir + 'vol2surf/' + sub + '_' + os.path.basename(run),'.nii.gz','_'+ hemi+'.gii')
        # vol2surf takes a while, so if we already saved the surface we can load it instead
        if os.path.exists(brain_bold_path) and not clobber:
            print(brain_bold_path + ' exists, loading...')
            brain_bold = surface.load_surf_data(brain_bold_path)
        else:
            print('Converting volume to surface for run: ' + brain_bold_path)
            # syntax for vol2surf using nilearn
            # lhbold = surface.vol_to_surf(img=bold,surf_mesh=surf, radius = 3)
            # use nsd precomputed mapping for vol2surf
            # outputfile = os.path.join(analysisdir, 'sub01_' + targetspace + '_' + str.replace(os.path.basename(runs[0]),'.nii.gz','.mgz'))

            sourcedata = run
            subjix = int(sub[-1])
            brain_bold = nsd.fit(
                subjix,
                sourcespace,
                targetspace,
                sourcedata,
                interptype='cubic',
                badval=0,
                outputfile=None,
                outputclass=None,
                fsdir=os.path.join(fsdir, sub))

            print('Saving as gii')
           # nib.freesurfer.io.write_morph_data(datadir + 'TEST.gii',run_data)
            img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(brain_bold)])
            nib.save(img, brain_bold_path)
            
        print('Vertices by Timepoints: ')
        print(brain_bold.shape)
        if brain_bold.shape[0] < brain_bold.shape[1]:
            brain_bold = np.transpose(brain_bold)
        print('Vertices by Timepoints')
        print(brain_bold.shape)
        brain_bold=np.delete(brain_bold,droptp,axis=1)
        
        # load motion regressors
        # TBD - this section can be removed because now the motion regressors are saved within the confound file
#         # We dont need to load the motion file at all, saving here in case we change the pipeline
#         motion_file = str.replace(run,'timeseries','motion')
#         motion_file = str.replace(motion_file,'.nii.gz','.tsv')
#         print(motion_file)
#         if os.path.exists(motion_file):
#             print('Loading motion regressors: '+ motion_file)
#             motion=pd.read_csv(motion_file,sep='\t',header=None)
#         else:
#             print('Cant locate: ' + motion_file)
#             continue
        
#         motion = motion.drop(droptp)
        
 
        # Now load second hemisphere if different from seed hemisphere
        if seed_hemi == hemi:
            print('same hemisphere used for seed and whole brain\n')
            seed_bold = brain_bold.copy()
        else:
            seed_bold_path = str.replace(analysisdir + 'vol2surf/' + sub + '_' + os.path.basename(run),'.nii.gz','_'+ seed_hemi+'.gii')
            print('Loading seed hemi '+seed_bold_path)
            # vol2surf takes a while, so if we already saved the surface we can load it instead
            if os.path.exists(seed_bold_path) and not clobber:
                print(seed_bold_path + ' exists, loading...')
                seed_bold = surface.load_surf_data(seed_bold_path)
            else:
                print('Converting volume to surface for run: ' + seed_bold_path)

                sourcedata = run
                subjix = int(sub[-1])
                seed_bold = nsd.fit(
                    subjix,
                    sourcespace,
                    seed_targetspace,
                    sourcedata,
                    interptype='cubic',
                    badval=0,
                    outputfile=None,
                    outputclass=None,
                    fsdir=os.path.join(fsdir, sub))

                print('Saving as gii')
               # nib.freesurfer.io.write_morph_data(datadir + 'TEST.gii',run_data)
                img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(seed_bold)])
                nib.save(img, seed_bold_path)

            print('Vertices by Timepoints for seed hemi: ')
            print(seed_bold.shape)
            if seed_bold.shape[0] < seed_bold.shape[1]:
                seed_bold = np.transpose(seed_bold)
            print('Vertices by Timepoints for seed hemi:')
            print(seed_bold.shape)
            seed_bold=np.delete(seed_bold,droptp,axis=1)

            
        # Denoise the data:
        brain_bold = np.transpose(signal.clean(signals = np.transpose(brain_bold), filter=myfilter,high_pass=0.008, low_pass = 0.1, 
                                       detrend=True, standardize='zscore',t_r=1.333, confounds =confounds))


        seed_bold = np.transpose(signal.clean(signals = np.transpose(seed_bold), filter=myfilter,high_pass=0.008, low_pass = 0.1, 
                                           detrend=True, standardize='zscore',t_r=1.333, confounds =confounds))


        # calculate timeseries for each seed ROI
        for roi in range(len(roi_names)):
            if new_labels:
                cur_roi_path = fssub + '/label/'+roi_names[roi] + '.label'
                if os.path.exists(cur_roi_path):
                    print('Loading ROI: ' + cur_roi_path)
                    cur_roi = surface.load_surf_data(cur_roi_path)
                    cur_roi = cur_roi.astype(int)
                    print('Label size:')
                    print(cur_roi.shape)
                    seed_timeseries = np.nanmean(seed_bold[cur_roi],axis=0)
                else:
                    print('Cant locate: ' + cur_roi_path)
                    continue
            else:
                seed_timeseries = np.nanmean(seed_bold[labels==roi_labels[roi]],axis=0)
 #           seed_timeseries_c = np.nanmean(lhbold_c[labels==roi_labels[roi]],axis=0)        
        # to plot
           # fig, ax = plt.subplots(figsize =(4, 3))
           # ax.plot(seed_timeseries)
           # ax.plot(seed_timeseries_c)
           # ax.set_title('seed timeseries confounds' + sub + ' ' + run)
           # ax.set_xlabel('Volume number')
           # ax.set_ylabel('Normalized signal')
           # print()

        
            roi_rval = np.empty(brain_bold.shape[0])
            roi_rval[:] = np.NaN
            for v in range(brain_bold.shape[0]):
                roi_rval[v]=np.corrcoef(np.transpose(seed_timeseries), np.transpose(brain_bold[v]))[0,1]

            roi_zmap[:,r,roi] = np.arctanh(roi_rval)

    #sub_filepath = analysisdir + sub + '_' + roi_names[roi] + '_' + str(len(runs)) + 'runs_' + hemi 
    sub_filepath = outputdir + sub + '_' + str(len(runs)) + 'runs_' + hemi 
    if faces==True:
        sub_filepath = outputdir + sub + '_' + str(len(runs)) + 'runs_faces_' + hemi 
    
    np.save(sub_filepath,roi_zmap)

    # for plotting
    if hemi == 'lh':
        h = 'left'
    elif hemi == 'rh':
        h = 'right'
    
    for roi in range(len(roi_names)):
        plotting.plot_surf_stat_map(surf, stat_map=np.nanmean(roi_zmap[:,:,roi],axis=1),
        hemi=h, threshold = .2, vmax=0.7, view='lateral', colorbar=True,
        bg_map=curv,title=sub + '_' + roi_names[roi] + '_' + str(len(runs)) + 'runs', output_file = sub_filepath + '_' + roi_names[roi] + '_lat.png')
        print()




