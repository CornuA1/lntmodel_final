clear;

loc_info = 'Q:\Documents\Harnett UROP\';
MOUSE = 'LF191022_1';
SESSION = ['20191209';'20191208'];
save_small_nm = '12';

cellreg_data = load(strcat(loc_info, MOUSE, '\', SESSION(1,:), '\prunned_cell_bodies.mat'));
cell_map = cellreg_data.res_log;
ind_lip = size(cell_map,1);

numb = size(SESSION,1);
x = 1;
for i = 1:ind_lip
    shot_mat = [];
    long_mat = [];
    shot_mat_ol = [];
    long_mat_ol = [];    
        cur_sess = SESSION(x,:);
        if cell_map(i,1) == 1
        file = strcat(loc_info, MOUSE, '\', cur_sess, '\data_glm_fit\',int2str(i),'.mat');
        data = load(file);
        if x == 1
            shot_mat = data.coef_short(2:end);
            long_mat = data.coef_long(2:end);
        else
            shot_mat_ol = data.coef_short(2:end);
            long_mat_ol = data.coef_long(2:end);
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
    

    
end

save_short = sum(result_short.');
save_long = sum(result_long.');


disp('Done!');

save(strcat(loc_info, MOUSE, '\',MOUSE,'_cells_',save_small_nm,'_histo.mat'),'save_short','save_long')

coef_matrix_short = short_total;
coef_matrix_long = long_total;


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

save(strcat(loc_info, MOUSE, '\',MOUSE,'_cells_',save_small_nm,'_matrix.mat'),'coef_matrix_short','coef_matrix_long')






