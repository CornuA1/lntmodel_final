function createHistogramFile(FILE, MOUSE)
%   save(save_loc,'r2_score','r2_ave','coef','coef_ave','execOrder')
 
disp(['Loading ',FILE,' and ',MOUSE]);
 
% load data
processed_data_path = strcat(FILE, MOUSE);
loaded_data = load(processed_data_path);
dF_ds = loaded_data.calcium_dF.';
ROI_amount = size(dF_ds,2);
 
pre_check_list = load(strcat(FILE, 'prunned_cell_bodies.mat'));
check_list = pre_check_list.res_log;
 
coefAves = zeros(52,sum(check_list));
r2Aves = zeros(1,sum(check_list));
count = 1;
 
for ROI=1:ROI_amount
    if check_list(ROI)
        
        glmFilePath = strcat(FILE,'glm_fits_calcium_gaussian\',int2str(ROI),'.mat');
        glmFile = load(glmFilePath);
        coefAves(:,count) = glmFile.coef_ave(1,2:end);
        r2Aves(:,count) = glmFile.r2_ave;
        count = count + 1;
    end
end
 
aveCoefAll = mean(coefAves,2);
aver2All = mean(r2Aves,2);
 
save_loc = strcat(FILE,'summaryFile.mat');
save(save_loc,'aveCoefAll','coefAves','r2Aves','aver2All')
 
disp(['Done with ',FILE,' and ',MOUSE]);
 
end


