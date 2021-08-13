clear;

cellreg_data = load('D:\Lukas\roi results\cellRegistered_LF191023_blue_layer_23_19_reg_ol.mat');
cell_map = cellreg_data.cell_registered_struct.cell_to_index_map;
ind_lis = size(cell_map,1);
ind_lip = size(cell_map,2);

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = 'LF191023_blue';
SESSION = ['20191119___';'20191119_ol'];
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
    cur_vet = comb_mat(i,:);
    shot_mat = [];
    long_mat = [];
    for x = 1:numb
        if x == 1
            cur_sess = SESSION(x,1:8);
        else
            cur_sess = SESSION(x,:);
        end
        file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', cur_sess, '\data_glm\',int2str(comb_mat(i,x)),'.mat');
        data = load(file);
        if x == 1
            shot_mat = data.coef_short(2:end);
            long_mat = data.coef_long(2:end);
        else
            shot_mat_ol = data.coef_short(2:end);
            long_mat_ol = data.coef_long(2:end);
        end
    end
    if i == 1
        short_total = shot_mat;
        long_total = long_mat;
        short_total_ol = shot_mat_ol;
        long_total_ol = long_mat_ol;
    else
        short_total = [short_total, shot_mat];
        long_total = [long_total, long_mat];
        short_total_ol = [short_total_ol, shot_mat];
        long_total_ol = [long_total_ol, long_mat];
    end
end

    short_mean = mean(short_total,2);
    makeFigure2(1,1,short_mean,1,1);
    long_mean = mean(long_total,2);
    makeFigure2(1,1,long_mean,1,1);
    
    short_mean_t = mean(short_total_ol,2);
    makeFigure2(1,1,short_mean_t,1,1);
    long_mean_t = mean(long_total_ol,2);
    makeFigure2(1,1,long_mean_t,1,1);
