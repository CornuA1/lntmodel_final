clear;

%cellreg_data = load('D:\Lukas\roi results\cellRegistered_LF191022_1_layer_5_2_3_ol.mat');
%cellreg_data = load('D:\Lukas\roi results\cellRegistered_LF191022_1_Dec_11_17_layer_5.mat');
%cellreg_data = load('D:\Lukas\roi results\cellRegistered_LF191022_1_Nov_14_Dec_09.mat');
cellreg_data = load('D:\Lukas\roi results\cellRegistered_LF191023_blue_13_08_reg.mat');
cell_map = cellreg_data.cell_registered_struct.cell_to_index_map;
ind_lis = size(cell_map,1);
ind_lip = size(cell_map,2);

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = 'LF191023_blue';
layer = 'naive';
%SESSION = ['20191204';'20191206';'20191208';'20191210'];
%SESSION = ['20191114';'20191115';'20191121';'20191125';'20191204';'20191207';'20191209'];
%SESSION = ['20191114';'20191114_ol';'20191115';'20191115_ol';'20191121';'20191121_ol';'20191125';'20191125_ol';'20191204';'20191204_ol';'20191207';'20191207_ol';'20191209';'20191209_ol'];
%SESSION = ['20191212';'20191215';'20191217'];
SESSION = ['20191113';'20191208'];
numb = size(SESSION,1);
comb_mat = [];

for i = 1:ind_lis
    count = 0;
    new_vect = cell_map(i,:);
    for x = 1:ind_lip
        if new_vect(x) == 0
            count = count + 1;
            break
        end
    end
    if count == 0
        if size(comb_mat) == 0
            comb_mat = new_vect;
        else
            comb_mat = [comb_mat; new_vect];
        end
    end
end

for i = 1:size(comb_mat,1)
%for i = 1:1
    cur_vet = comb_mat(i,:);
    shot_mat = [];
    long_mat = [];
    for x = 1:numb
        file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION(x,:), '\data_glm\',int2str(comb_mat(i,x)),'.mat');
        data = load(file);
        if x == 1
            shot_mat = data.coef_short;
            long_mat = data.coef_long;
        else
            shot_mat = [shot_mat, data.coef_short];
            long_mat = [long_mat, data.coef_long];
        end
    end
    [Rs,Ps] = corrcoef(shot_mat);
    [Rl,Pl] = corrcoef(long_mat);
    if i == 1
        RS = Rs;
        RL = Rl;
    else
        RS = RS + Rs;
        RL = RL + Rl;
    end
end

hs = RS/size(comb_mat,1);
hl = RL/size(comb_mat,1);
save(strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', MOUSE,layer,'.mat'),'hs','hl')
