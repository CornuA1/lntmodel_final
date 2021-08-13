clear;

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = ['LF191022_1___';'LF191022_3___';'LF191023_blue'];
MOUSE = MOUSE(3,1:end);
SESSION = ['20191115';'20191209';'20191113';'20191207';'20191113';'20191208';'20191217';'20191215'];
SESSION = SESSION(8,:);
SESS_NUM = ['M01_000_004';'M01_000_000';'M01_000_000';'M01_000_002';'M01_000_003';'M01_000_002';];
SESS_NUM = SESS_NUM(5,:);
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
%{
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
%}

result_short = zeros(size(coef_matrix_short,1),size(coef_matrix_short,2));
result_long = zeros(size(coef_matrix_long,1),size(coef_matrix_long,2));

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

save(strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION,'\',SESS_NUM,'_new_histo.mat'),'save_short','save_long')



