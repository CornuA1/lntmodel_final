clear;

cellreg_data = load('D:\Lukas\roi results\cellRegistered_LF191023_blue_04_08_10_reg');
cell_map = cellreg_data.cell_registered_struct.cell_to_index_map;
ind_lis = size(cell_map,1);
ind_lip = size(cell_map,2);

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = 'LF191023_blue';
SESSION = ['20191204';'20191208';'20191210'];
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
        cur_sess = SESSION(x,:);
        file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', cur_sess, '\data_glm_poisson_c\',int2str(comb_mat(i,x)),'.mat');
        data = load(file);
        if x == 1
            shot_mat = data.coef_short(2:end);
            long_mat = data.coef_long(2:end);
        elseif x == 2
            shot_mat_ol = data.coef_short(2:end);
            long_mat_ol = data.coef_long(2:end);
        else
            shot_mat_3 = data.coef_short(2:end);
            long_mat_3 = data.coef_long(2:end);
        end
    end
    if i == 1
        short_total = shot_mat;
        long_total = long_mat;
        short_total_ol = shot_mat_ol;
        long_total_ol = long_mat_ol;
        short_total_3 = shot_mat_3;
        long_total_3 = long_mat_3;        
    else
        short_total = [short_total, shot_mat];
        long_total = [long_total, long_mat];
        short_total_ol = [short_total_ol, shot_mat_ol];
        long_total_ol = [long_total_ol, long_mat_ol];
        short_total_3 = [short_total_3, shot_mat_3];
        long_total_3 = [long_total_3, long_mat_3];          
    end
end

coef_matrix_short = short_total;
coef_matrix_long = long_total;
coef_matrix_short_x = short_total_ol;
coef_matrix_long_x = long_total_ol;
coef_matrix_short_x3 = short_total_3;
coef_matrix_long_x3 = long_total_3;

result_short = zeros(size(coef_matrix_short,1),size(coef_matrix_short,2));
result_long = zeros(size(coef_matrix_long,1),size(coef_matrix_long,2));
result_short_x = zeros(size(coef_matrix_short_x,1),size(coef_matrix_short_x,2));
result_long_x = zeros(size(coef_matrix_long_x,1),size(coef_matrix_long_x,2));
result_short_x3 = zeros(size(coef_matrix_short_x3,1),size(coef_matrix_short_x3,2));
result_long_x3 = zeros(size(coef_matrix_long_x3,1),size(coef_matrix_long_x3,2));

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
    
    max_short_x3 = max(coef_matrix_short_x3(:,x));
    if max_short_x3 > 0
        elm_s_1_x3 = find(coef_matrix_short_x3(:,x)==max_short_x3);
        result_short_x3(elm_s_1_x3,x) = 1;
        coef_matrix_short_x3(elm_s_1_x3,x) = 0;
    end
    
    max_long_x3 = max(coef_matrix_long_x3(:,x));
    if max_long_x3 > 0
        elm_l_1_x3 = find(coef_matrix_long_x3(:,x)==max_long_x3);
        result_long_x3(elm_l_1_x3,x) = 1;
        coef_matrix_long_x3(elm_l_1_x3,x) = 0;
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
    
    max_short_x3 = max(coef_matrix_short_x3(:,x));
    if max_short_x3 > 0
        elm_s_1_x3 = find(coef_matrix_short_x3(:,x)==max_short_x3);
        result_short_x3(elm_s_1_x3,x) = 1;
        coef_matrix_short_x3(elm_s_1_x3,x) = 0;
    end
    
    max_long_x3 = max(coef_matrix_long_x3(:,x));
    if max_long_x3 > 0
        elm_l_1_x3 = find(coef_matrix_long_x3(:,x)==max_long_x3);
        result_long_x3(elm_l_1_x3,x) = 1;
        coef_matrix_long_x3(elm_l_1_x3,x) = 0;
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
    
    max_short_x3 = max(coef_matrix_short_x3(:,x));
    if max_short_x3 > 0
        elm_s_1_x3 = find(coef_matrix_short_x3(:,x)==max_short_x3);
        result_short_x3(elm_s_1_x3,x) = 1;
        coef_matrix_short_x3(elm_s_1_x3,x) = 0;
    end
    
    max_long_x3 = max(coef_matrix_long_x3(:,x));
    if max_long_x3 > 0
        elm_l_1_x3 = find(coef_matrix_long_x3(:,x)==max_long_x3);
        result_long_x3(elm_l_1_x3,x) = 1;
        coef_matrix_long_x3(elm_l_1_x3,x) = 0;
    end    
    
end

save_short = sum(result_short.');
save_long = sum(result_long.');
save_short_x = sum(result_short_x.');
save_long_x = sum(result_long_x.');
save_short_x3 = sum(result_short_x3.');
save_long_x3 = sum(result_long_x3.');

disp('Done!');

save(strcat(loc_info.raw_dir(2:end-1), MOUSE, '\',MOUSE,'_coef_matched_23_x_histo.mat'),'save_short','save_long','save_short_x','save_long_x',...
    'save_short_x3','save_long_x3')
%{
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

save(strcat(loc_info.raw_dir(2:end-1), MOUSE, '\',MOUSE,'_coef_matched_5_x_heatmap.mat'),'coef_matrix_short','coef_matrix_long','coef_matrix_short_x','coef_matrix_long_x')

%}


