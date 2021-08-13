clear;

rootdir = 'D:\Lukas\data\animals_raw\LF191022_1\20191207\';
ROI = 1;

filelist = dir(fullfile(rootdir, '**\*_results.mat'));
roi = load([rootdir,'glm_fits_calcium_gaussian\',int2str(ROI),'.mat']);
fit_dist = 'gaussian';
processed_data_path = strcat(rootdir, filelist(1).name);
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';
dF_ds = loaded_data.calcium_dF.';
glmFileDir = strcat(rootdir, 'GLMregs.mat');
glmFiles = load(glmFileDir);
glmFiles = glmFiles.regressors;
options = glmnetSet();
options.lambda_min = 0.2;
options.alpha = 0.5;
disp(int2str(ROI));
sigUse = dF_ds(:,ROI);
for val=1:length(roi.execOrder)
    valueTrial = roi.execOrder(val);
    blackTrial = valueTrial+1;
    trialInReg = behav_ds(:,7) == valueTrial; % Add the leading blackbox
    trialInBlack = behav_ds(:,7) == blackTrial;
    trialIn = logical(trialInReg + trialInBlack);
    trialEx = not(trialIn);
    out = glmRun(glmFiles(trialEx,:), glmFiles(trialIn,:), sigUse(trialEx,:), sigUse(trialIn,:), fit_dist, options);
end