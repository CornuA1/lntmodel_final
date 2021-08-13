clear;

loc_info = 'D:\Lukas\data\animals_raw\';
MOUSE = 'LF191022_1';
SESSION = ['20191115';'20191209'];
save_small_nm = '15_09';

cellreg_data = load(strcat(loc_info, MOUSE, '\', SESSION(1,:), '\matched_cells.mat'));
cell_map = cellreg_data.new_res;
ind_lip = size(cell_map,1);

trans_1_data = load(strcat(loc_info, MOUSE, '\', SESSION(1,:),'\cell_transfer_map.mat'));
trans_1 = trans_1_data.trans;
trans_2_data = load(strcat(loc_info, MOUSE, '\', SESSION(2,:), '\cell_transfer_map.mat'));
trans_2 = trans_2_data.trans;

numb = size(SESSION,1);

for i = 1:ind_lip
    shot_mat = [];
    long_mat = [];
    for x = 1:numb
        cur_sess = SESSION(x,:);
        if x == 1
            trans = trans_1;
        else
            trans = trans_2;
        end
        file = strcat(loc_info, MOUSE, '\', cur_sess, '\glm_fits_calcium_gaussian\',int2str(find(trans(:,3) == cell_map(i,x))),'.mat');
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
        short_total_ol = [short_total_ol, shot_mat_ol];
        long_total_ol = [long_total_ol, long_mat_ol];
    end
end

coef_matrix_short = short_total;
coef_matrix_long = long_total;
coef_matrix_short_x = short_total_ol;
coef_matrix_long_x = long_total_ol;

result_short = zeros(size(coef_matrix_short,1),size(coef_matrix_short,2));
result_long = zeros(size(coef_matrix_long,1),size(coef_matrix_long,2));
result_short_x = zeros(size(coef_matrix_short_x,1),size(coef_matrix_short_x,2));
result_long_x = zeros(size(coef_matrix_long_x,1),size(coef_matrix_long_x,2));

for x = 1:size(coef_matrix_short,2)
    max_short = max(coef_matrix_short(:,x));
    if max_short > 0
        elm_s_1 = find(coef_matrix_short(:,x)==max_short);
        result_short(elm_s_1,x) = 1;
        coef_matrix_short(elm_s_1,x) = 0;
    end
    
    max_long = max(coef_matrix_long(:,x));
    if max_long > 0
        elm_l_1 = find(coef_matrix_long(:,x)==max_long);
        result_long(elm_l_1,x) = 1;
        coef_matrix_long(elm_l_1,x) = 0;
    end
    
    max_short_x = max(coef_matrix_short_x(:,x));
    if max_short_x > 0
        elm_s_1_x = find(coef_matrix_short_x(:,x)==max_short_x);
        result_short_x(elm_s_1_x,x) = 1;
        coef_matrix_short_x(elm_s_1_x,x) = 0;
    end
    
    max_long_x = max(coef_matrix_long_x(:,x));
    if max_long_x > 0
        elm_l_1_x = find(coef_matrix_long_x(:,x)==max_long_x);
        result_long_x(elm_l_1_x,x) = 1;
        coef_matrix_long_x(elm_l_1_x,x) = 0;
    end
    
end

for x = 1:size(coef_matrix_short,2)
    max_short = max(coef_matrix_short(:,x));
    if max_short > 0
        elm_s_1 = find(coef_matrix_short(:,x)==max_short);
        result_short(elm_s_1,x) = 1;
        coef_matrix_short(elm_s_1,x) = 0;
    end
    
    max_long = max(coef_matrix_long(:,x));
    if max_long > 0
        elm_l_1 = find(coef_matrix_long(:,x)==max_long);
        result_long(elm_l_1,x) = 1;
        coef_matrix_long(elm_l_1,x) = 0;
    end
    
    max_short_x = max(coef_matrix_short_x(:,x));
    if max_short_x > 0
        elm_s_1_x = find(coef_matrix_short_x(:,x)==max_short_x);
        result_short_x(elm_s_1_x,x) = 1;
        coef_matrix_short_x(elm_s_1_x,x) = 0;
    end
    
    max_long_x = max(coef_matrix_long_x(:,x));
    if max_long_x > 0
        elm_l_1_x = find(coef_matrix_long_x(:,x)==max_long_x);
        result_long_x(elm_l_1_x,x) = 1;
        coef_matrix_long_x(elm_l_1_x,x) = 0;
    end
    
end

for x = 1:size(coef_matrix_short,2)
    max_short = max(coef_matrix_short(:,x));
    if max_short > 0
        elm_s_1 = find(coef_matrix_short(:,x)==max_short);
        result_short(elm_s_1,x) = 1;
        coef_matrix_short(elm_s_1,x) = 0;
    end
    
    max_long = max(coef_matrix_long(:,x));
    if max_long > 0
        elm_l_1 = find(coef_matrix_long(:,x)==max_long);
        result_long(elm_l_1,x) = 1;
        coef_matrix_long(elm_l_1,x) = 0;
    end
    
    max_short_x = max(coef_matrix_short_x(:,x));
    if max_short_x > 0
        elm_s_1_x = find(coef_matrix_short_x(:,x)==max_short_x);
        result_short_x(elm_s_1_x,x) = 1;
        coef_matrix_short_x(elm_s_1_x,x) = 0;
    end
    
    max_long_x = max(coef_matrix_long_x(:,x));
    if max_long_x > 0
        elm_l_1_x = find(coef_matrix_long_x(:,x)==max_long_x);
        result_long_x(elm_l_1_x,x) = 1;
        coef_matrix_long_x(elm_l_1_x,x) = 0;
    end
    
end

save_short = sum(result_short.');
save_long = sum(result_long.');
save_short_x = sum(result_short_x.');
save_long_x = sum(result_long_x.');

disp('Done!');

save(strcat(loc_info, MOUSE, '\',MOUSE,'_matched_cells_',save_small_nm,'_cal_histo.mat'),'save_short','save_long','save_short_x','save_long_x')

coef_matrix_short = short_total;
coef_matrix_long = long_total;
coef_matrix_short_x = short_total_ol;
coef_matrix_long_x = long_total_ol;

for x = 1:size(coef_matrix_short,1)
    for y = 1:size(coef_matrix_short,2)
        if coef_matrix_short(x,y) < 0
            coef_matrix_short(x,y) = 0;
        end
        if coef_matrix_long(x,y) < 0
            coef_matrix_long(x,y) = 0;
        end
        if coef_matrix_long_x(x,y) < 0
            coef_matrix_long_x(x,y) = 0;
        end
        if coef_matrix_short_x(x,y) < 0
            coef_matrix_short_x(x,y) = 0;
        end
    end
end

for x = 1:size(coef_matrix_short,2)
    max_short = max(coef_matrix_short(:,x));
    coef_matrix_short(:,x) = coef_matrix_short(:,x)/max_short;
    max_long = max(coef_matrix_long(:,x));
    coef_matrix_long(:,x) = coef_matrix_long(:,x)/max_long;
    
    max_short_x = max(coef_matrix_short_x(:,x));
    coef_matrix_short_x(:,x) = coef_matrix_short_x(:,x)/max_short_x;
    max_long_x = max(coef_matrix_long_x(:,x));
    coef_matrix_long_x(:,x) = coef_matrix_long_x(:,x)/max_long_x;
end

save(strcat(loc_info, MOUSE, '\',MOUSE,'_matched_cells_',save_small_nm,'_cal_matrix.mat'),'coef_matrix_short','coef_matrix_long','coef_matrix_short_x','coef_matrix_long_x')






