function createGLMdata(FILE, MOUSE, data_type, fit_dist)

% data_type = {'calcium', 'detrended' , 'deconvolved'}
% fit_dist = {'gaussian', 'poisson'}

disp(['Loading ',FILE,' and ',MOUSE]);

% load data
processed_data_path = strcat(FILE, MOUSE);
loaded_data = load(processed_data_path);
behav_ds = loaded_data.behaviour_aligned.';
if strcmp(data_type,'calcium')
    dF_ds = loaded_data.calcium_dF.';
elseif strcmp(data_type,'detrended')
    dF_ds = loaded_data.deconv.';
elseif strcmp(data_type,'deconvolved')
    dF_ds = loaded_data.spikes.';
end
ROI_amount = size(dF_ds,2);

trial = (behav_ds(end,7) - 1)/2;
trials = 1:trial;
trials = trials*2-1;

glmFileDir = strcat(FILE, 'GLMregs.mat');
glmFiles = load(glmFileDir);
glmFiles = glmFiles.regressors;
pre_check_list = load(strcat(FILE, 'prunned_cell_bodies.mat'));
check_list = pre_check_list.res_log;

options = glmnetSet();
options.lambda_min = 0.2;
options.alpha = 0.5;

visit = false;
for ROI=1:ROI_amount
if check_list(ROI)
    disp(int2str(ROI));
    sigUse = dF_ds(:,ROI);
    if strcmp(fit_dist,'poisson')
        sigUse = sigUse - min(sigUse);
    end
    execOrder = trials(randperm(length(trials))); 
    
    r2_score = zeros(length(trials),1);
    coef = zeros(length(trials),53);
    
    for val=1:length(execOrder)
        valueTrial = execOrder(val);
        blackTrial = valueTrial+1;
        trialInReg = behav_ds(:,7) == valueTrial; % Add the leading blackbox
        trialInBlack = behav_ds(:,7) == blackTrial;
        trialIn = logical(trialInReg + trialInBlack);
        trialEx = not(trialIn);
        out = glmRun(glmFiles(trialEx,:), glmFiles(trialIn,:), sigUse(trialEx,:), sigUse(trialIn,:), fit_dist, options);
        r2_score(val,1) = out.r2;
        coef(val,:) = out.coef;
    end
    
    r2_ave = mean(r2_score);
    coef_ave = mean(coef,1);
    

    save_loc = strcat(FILE,'glm_fits_',data_type,'_',fit_dist,'\',int2str(ROI),'.mat');
    if ~visit
        status = mkdir(strcat(FILE,'glm_fits_',data_type,'_',fit_dist));
        visit = true;
    end
    save(save_loc,'r2_score','r2_ave','coef','coef_ave','execOrder')

end
end

disp(['Done with ',FILE,' and ',MOUSE]);

end