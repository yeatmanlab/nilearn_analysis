% HBN_DownloadFiles
% This script loads a csv with a list of subjects and downloads their
% preprocessed data from S3

% TODO
% 1. Add test to see if local folder already exist to avoid copying
%    twice
% 2. Consider changing target file names to include subject name
% 3. Think about a way to check for mix ups between subjects

clear
downloadScans = 1;
downloadMovmentRegressors = 1;
downloadFD = 1;
task = 'rest'; % 'rest' or 'movie'

% Load csv file with list of subjects, generated in R
%data_file_name = 'subs_preprocessed_qc_fullphen_EQ70_meanFD40q_SNR75q_WIAT.csv'; % N=53
data_file_name = 'subs_preprocessed_restingstate_movieDM_meanFD03_SNR04_WIAT.csv'; % N=67, with movie, not filtered for diffusion QC or handedness
data_file_name = 'subs_preprocessed_restingstate_movieDM_meanFD05_SNR15_FD05_90_WIAT.csv'; %N=92
data_file_name = 'subs_preprocessed_onlyrest_meanFD05_SNR15_FD05_90_WIAT.csv'; % N=216
#data_file_name = 'subs_preprocessed_onlymovie_meanFD05_SNR15_FD05_90_WIAT.csv'; % N=120
data_file = readtable(fullfile('/scratch/groups/jyeatman/HBN_FC',data_file_name));
participant_list = data_file.('participant_id');

% Define file names and path names
s3_folder        = 's3://fcp-indi/data/Projects/HBN/CPAC_preprocessed_Derivatives';
%local_folder     = '/mnt/disks/brain/MayasProjects/Data/HBN/CPAC_Preprocessed';
local_folder     = '/scratch/groups/jyeatman/HBN_FC/input';
anatPath = 'anatomical_to_standard/transform_Warped.nii.gz'; % anat path is the same for all scans

switch task
    case 'rest'
        AllFiles = {'bandpassed_demeaned_filtered_antswarp.nii.gz',...
            'bandpassed_demeaned_filtered_antswarp.nii.gz'};
        AllPaths = {'functional_to_standard/_scan_rest_run-1/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5',...
            'functional_to_standard/_scan_rest_run-2/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5'};
        movementFiles = {'_task-rest_run-1_bold_calc_resample.1D',...
            '_task-rest_run-2_bold_calc_resample.1D'};
        movementPaths = {'movement_parameters/_scan_rest_run-1',...
            'movement_parameters/_scan_rest_run-2'};
        FDPaths = {'frame_wise_displacement_power/_scan_rest_run-1',...
            'frame_wise_displacement_power/_scan_rest_run-2'};
        
    case 'movie'
        AllFiles = {'bandpassed_demeaned_filtered_antswarp.nii.gz'};
        AllPaths = {'functional_to_standard/_scan_movieDM/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5'};
        movementFiles = {'_task-movieDM_bold_calc_resample.1D'};
        movementPaths = {'movement_parameters/_scan_movieDM'};
        FDPaths = {'frame_wise_displacement_power/_scan_movieDM'};
end


% Start looping throught participants, download files if they dont exist
for id = 1:numel(participant_list)
    if downloadScans
        % first download anatomical file
        anat_source = fullfile(s3_folder,[participant_list{id} '_ses-1'],anatPath);
        anat_target = fullfile(local_folder,participant_list{id},anatPath);
        if ~exist(anat_target,'file')
            cmd_str = ['aws s3 cp ' anat_source ' ' anat_target];
            try
                system(cmd_str);
            catch
                fprintf(1,['Error downloading ' anat_source ' \n']);
            end
        else
            fprintf(1,[anat_target ' already exists in path\n']);
        end

    for fileID = 1:numel(AllFiles)
        file_source = fullfile(s3_folder,[participant_list{id} '_ses-1'],AllPaths{fileID},AllFiles{fileID});
        file_target = fullfile(local_folder,participant_list{id},AllPaths{fileID},AllFiles{fileID});
        file_target_renamed = fullfile(local_folder,participant_list{id},AllPaths{fileID},['movieDM_' AllFiles{fileID}]);
        if ~exist(file_target,'file')
            if ~exist(file_target_renamed,'file')
                cmd_str = ['aws s3 cp ' file_source ' ' file_target];
                try
                    system(cmd_str);
                    participant_list{id,fileID+1} = 1;
                catch
                    fprintf(1,['Error downloading ' file_source ' \n']);
                    participant_list{id,fileID+1} = NaN;
                end
            else
                fprintf(1,[file_target_renamed ' already exists in path\n']);
                participant_list{id,fileID+1} = 0;
            end
        else
            fprintf(1,[file_target ' already exists in path\n']);
            participant_list{id,fileID+1} = 0;
        end
    end
end

% Download and rename movement files - replace 1D with 1d
clear fileID
if downloadMovmentRegressors
    for fileID = 1:numel(movementFiles)
        file_source = fullfile(s3_folder,[participant_list{id} '_ses-1'],movementPaths{fileID},[participant_list{id} movementFiles{fileID}]);
        file_target = fullfile(local_folder,participant_list{id},movementPaths{fileID},[participant_list{id} strrep(movementFiles{fileID},'.1D','.1d')]);
        if ~exist(file_target,'file')
            cmd_str = ['aws s3 cp ' file_source ' ' file_target];
            try
                system(cmd_str);
                participant_list{id,fileID+8} = 1;
            catch
                fprintf(1,['Error downloading ' file_source ' \n']);
                participant_list{id,fileID+8} = NaN;
            end
        else
            fprintf(1,[file_target ' already exists in path\n']);
            participant_list{id,fileID+8} = 0;
        end
    end
end

% Download and rename FD files - replace 1D with 1d
clear fileID
if downloadFD
    for fileID = 1:numel(FDPaths)
        file_source = fullfile(s3_folder,[participant_list{id} '_ses-1'],FDPaths{fileID},'FD.1D');
        file_target = fullfile(local_folder,participant_list{id},FDPaths{fileID},'FD.1d');
        if ~exist(file_target,'file')
            cmd_str = ['aws s3 cp ' file_source ' ' file_target];
            try
                system(cmd_str);
                participant_list{id,fileID+6} = 1;
            catch
                fprintf(1,['Error downloading ' file_source ' \n']);
                participant_list{id,fileID+6} = NaN;
            end
        else
            fprintf(1,[file_target ' already exists in path\n']);
            participant_list{id,fileID+6} = 0;
        end
    end
end
end


% Rename movie files - add 'movieDM' to the file beginning
if strcmp(task,'movie')
    moviePath = AllPaths{1}; % HARD-CODED!!!
    origFileName = 'bandpassed_demeaned_filtered_antswarp.nii.gz';
    newFileName = ['movieDM_' origFileName];
    for ii = 1:length(participant_list)
        oldname = fullfile(local_folder,participant_list{ii},moviePath,origFileName);
        newname = strrep(oldname,origFileName,newFileName);
        if exist(oldname,'file')
            movefile(oldname, newname);
        end
    end
end
% % Test
% % Loop on all downloaded folders, open niftis and check number of volumes
% % per scan
% subjects_volumes = zeros(numel(participant_list),numel(AllFiles));
% for id = 1:numel(participant_list)
%     for fileID = 1:numel(AllFiles)
%         file_target = fullfile(local_folder,participant_list{id},AllPaths{fileID},AllFiles{fileID});
%         if ~exist(file_target, 'file')
%             file_target = fullfile(local_folder,participant_list{id},AllPaths{fileID},['movieDM_' AllFiles{fileID}]);
%         end
%         if exist(file_target,'file')
%             cur_nifti = niftiRead(file_target);
%             cur_dim = size(cur_nifti.data);
%             if length(cur_dim) > 3
%                 subjects_volumes(id,fileID) = cur_dim(4);
%             else
%                 subjects_volumes(id,fileID) = 0;
%             end
%             clear cur_nifti
%         end
%     end
% end
%
% test = 1;
