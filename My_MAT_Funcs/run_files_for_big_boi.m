clear;

loc_info = 'Q:\Documents\Harnett UROP\';
MOUSE = 'LF191024_1';
SESSION = ['20191114';'20191210'];
save_small_nm = '14_10';

cellreg_data = load(strcat(loc_info, MOUSE, '\', SESSION(2,:), '\matched_cells.mat'));
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
        file = strcat(loc_info, MOUSE, '\', cur_sess, '\data_glm_fit\',int2str(find(trans(:,3) == cell_map(i,x))),'.mat');
        data = load(file);
        if x == 1
            shot_mat = data.coef_short(2:end);
            long_mat = data.coef_long(2:end);
            for y = 1:length(shot_mat)
                if shot_mat(y) < 0
                    shot_mat(y) = 0;
                end
                if long_mat(y) < 0
                    long_mat(y) = 0;
                end
            end
        else
            shot_mat_ol = data.coef_short(2:end);
            long_mat_ol = data.coef_long(2:end);
            for y = 1:length(shot_mat_ol)
                if shot_mat_ol(y) < 0
                    shot_mat_ol(y) = 0;
                end
                if long_mat_ol(y) < 0
                    long_mat_ol(y) = 0;
                end
            end            
        end
    end
    if i == 1
        naive_total = shot_mat + long_mat;
        expert_total = shot_mat_ol + long_mat_ol;
    else
        naive_total = naive_total + shot_mat + long_mat;
        expert_total = expert_total + shot_mat_ol + long_mat_ol;
    end
end



save(strcat(loc_info, MOUSE, '\',MOUSE,'_big_boi_',save_small_nm,'.mat'),'naive_total','expert_total')
