function createMatchedHistograms(MOUSE, session1, session2)
loc_info = 'D:\Lukas\data\animals_raw\';
SESSION = [session1;session2];

cellreg_data = load(strcat(loc_info, MOUSE, '\', SESSION(1,:), '\matched_cells.mat'));
cell_map = cellreg_data.new_res;
ind_lip = size(cell_map,1);

trans_1_data = load(strcat(loc_info, MOUSE, '\', SESSION(1,:),'\cell_transfer_map.mat'));
trans_1 = trans_1_data.trans;
trans_2_data = load(strcat(loc_info, MOUSE, '\', SESSION(2,:), '\cell_transfer_map.mat'));
trans_2 = trans_2_data.trans;

numb = size(SESSION,1);
r_short_naive = 0;
r_short_expert = 0;
count1 = 0;
count3 = 0;

for i = 1:ind_lip
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
            if ~isnan(data.r2_ave)
                r_short_naive = r_short_naive + data.r2_ave;
                count1 = count1 + 1;
            end
        else
            if ~isnan(data.r2_ave)
                r_short_expert = r_short_expert + data.r2_ave;
                count3 = count3 + 1;
            end
        end
    end
end

coefAvesNaive = zeros(52,count1);
r2AvesNaive = r_short_naive/count1;
countNaive = 1;

coefAvesExpert = zeros(52,count3);
r2AvesExpert = r_short_expert/count3;
countExpert = 1;

for i = 1:ind_lip
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
            if ~isnan(data.r2_ave)
                coefAvesNaive(:,countNaive) = data.coef_ave(1,2:end);
                countNaive = countNaive + 1;
            end
        else
            if ~isnan(data.r2_ave)
                coefAvesExpert(:,countExpert) = data.coef_ave(1,2:end);
                countExpert = countExpert + 1;
            end
        end
    end
end

save_loc = strcat(loc_info,MOUSE,'\summaryMatchFile.mat');
save(save_loc,'r2AvesNaive','r2AvesExpert','coefAvesNaive','coefAvesExpert')

disp(strcat('Done with---',MOUSE));
end