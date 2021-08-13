clear;

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = 'LF191022_1';
SESSION = ['20191114';'20191125';'20191204';'20191209';'20191213';'20191217'];
numb = size(SESSION,1);

r2_gau = zeros(numb,1);
r2_sig = zeros(numb,1);
r2_exp = zeros(numb,1);

for i = 1:size(SESSION,1)
    file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION(i,:), '\data_glm_fit\','1.mat');
    data = load(file);
    r2_gau(i) = data.r2_short;
    file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION(i,:), '\data_glm_fit_sig\','1.mat');
    data = load(file);
    r2_sig(i) = data.r2_short;
    file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION(i,:), '\data_glm_fit_exp\','1.mat');
    data = load(file);
    r2_exp(i) = data.r2_short;    
end