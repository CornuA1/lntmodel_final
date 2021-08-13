clear;

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = ['LF191022_1___';'LF191022_3___';'LF191023_blue'];
MOUSE = MOUSE(3,:);
SESSION_list= '201912';
SESSION = [SESSION_list,'17'];
SESS_NUM_base = 'M01_000_00';
SESS_NUM = [SESS_NUM_base,'6'];
file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION,'\',SESS_NUM,'_results.mat');
data = load(file);
num_roi = size(data.C_dec,1);
coef_matrix_short = [];
coef_matrix_long = [];


for i = 1:num_roi   
    file_roi = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\data_glm\',int2str(i),'.mat');
    data_roit = load(file_roi);
        if i == 1
            coef_matrix_short = data_roit.coef_short(2:end);
            coef_matrix_long = data_roit.coef_long(2:end);
        else
            coef_matrix_short = [coef_matrix_short data_roit.coef_short(2:end)];
            coef_matrix_long = [coef_matrix_long data_roit.coef_long(2:end)];
        end
end

for x = 1:size(coef_matrix_short,1)
    for y = 1:size(coef_matrix_short,2)
        if coef_matrix_short(x,y) < 0
            coef_matrix_short(x,y) = 0;
        end
        if coef_matrix_long(x,y) < 0
            coef_matrix_long(x,y) = 0;
        end
    end
end

for x = 1:size(coef_matrix_short,2)
    max_short = max(coef_matrix_short(:,x));
    coef_matrix_short(:,x) = coef_matrix_short(:,x)/max_short;
    max_long = max(coef_matrix_long(:,x));
    coef_matrix_long(:,x) = coef_matrix_long(:,x)/max_long;
end

coef_short = coef_matrix_short.';

coef_long = coef_matrix_long.';

save(strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION,'\',SESS_NUM,'_heatmaps.mat'),'coef_short','coef_long')



