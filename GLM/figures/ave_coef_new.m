clear;

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = 'LF191022_3';
SESSION = '20191207';
SESS_NUM = 'M01_000_002';
file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION,'\',SESS_NUM,'_results.mat');
data = load(file);
num_roi = size(data.C_dec,1);
coef_matrix_short = [];
coef_matrix_long = [];


for i = 1:num_roi   
    file_roi = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION, '\data_glm_poisson_c\',int2str(i),'.mat');
    data_roit = load(file_roi);
        if i == 1
            coef_matrix_short = data_roit.coef_short(2:end);
            coef_matrix_long = data_roit.coef_long(2:end);
        else
            coef_matrix_short = [coef_matrix_short data_roit.coef_short(2:end)];
            coef_matrix_long = [coef_matrix_long data_roit.coef_long(2:end)];
        end
end

coef_short = coef_matrix_short.';
short_mean = mean(coef_short);
short_std = std(coef_short);
coef_long = coef_matrix_long.';
long_mean = mean(coef_long);
long_std = std(coef_long);

x = 1:28;

figure;
pos2 = [0.1 0.4 0.8 0.25];
subplot('Position',pos2)
errorbar(x,short_mean,short_std,'ro');
xticks(x)
xticklabels({'-180','-160','-140','-120','-100','-80','-60','-40',...
    '-20','0','20','40','60','80','100','120','140','160',...
    'Slow Speed','Fast Speed','Linear Speed','Lick Location','Reward Event',...
    'Reward Event -30','Reward Event +30','Trial Onset','Trial Onset +30','Trial Onset +60'})
xtickangle(45)


figure;
pos2 = [0.1 0.4 0.8 0.25];
subplot('Position',pos2)
errorbar(x,long_mean,long_std,'ro');
xticks(x)
xticklabels({'-180','-160','-140','-120','-100','-80','-60','-40',...
    '-20','0','20','40','60','80','100','120','140','160',...
    'Slow Speed','Fast Speed','Linear Speed','Lick Location','Reward Event',...
    'Reward Event -30','Reward Event +30','Trial Onset','Trial Onset +30','Trial Onset +60'})
xtickangle(45)

