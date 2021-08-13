clear;

loc_info = 'Q:\Documents\Harnett UROP\';
MOUSE = 'LF191022_1';
SESSION = ['20191115';'20191209'];
save_small_nm = '15_09';

cellreg_data = load(strcat(loc_info, MOUSE, '\', SESSION(2,:), '\matched_cells.mat'));
cell_map = cellreg_data.new_res;
ind_lip = size(cell_map,1);

trans_1_data = load(strcat(loc_info, MOUSE, '\', SESSION(1,:),'\cell_transfer_map.mat'));
trans_1 = trans_1_data.trans;
trans_2_data = load(strcat(loc_info, MOUSE, '\', SESSION(2,:), '\cell_transfer_map.mat'));
trans_2 = trans_2_data.trans;

numb = size(SESSION,1);
r_short_naive = 0;
r_short_expert = 0;
r_long_naive = 0;
r_long_expert = 0;
count1 = 0;
count2 = 0;
count3 = 0;
count4 = 0;


for i = 1:ind_lip
    for x = 1:numb
        cur_sess = SESSION(x,:);
        if x == 1
            trans = trans_1;
        else
            trans = trans_2;
        end
        file = strcat(loc_info, MOUSE, '\', cur_sess, '\data_glm_fit_news\',int2str(find(trans(:,3) == cell_map(i,x))),'.mat');
        data = load(file);
        if x == 1
            if ~isnan(data.r2_short)
                r_short_naive = r_short_naive + data.r2_short;
                count1 = count1 + 1;
            end
            if ~isnan(data.r2_long)
                r_long_naive = r_long_naive + data.r2_long;
                count2 = count2 + 1;
            end
        else
            if ~isnan(data.r2_short)
                r_short_expert = r_short_expert + data.r2_short;
                count3 = count3 + 1;
            end
            if ~isnan(data.r2_long)
                r_long_expert = r_long_expert + data.r2_long;
                count4 = count4 + 1;
            end
        end
    end
end

n_r_short_naive = r_short_naive/count1;
n_r_short_expert = r_short_expert/count3;
n_r_long_naive = r_long_naive/count2;
n_r_long_expert = r_long_expert/count4;

disp('Donezo!');


