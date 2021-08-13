clear;

mouse = 'LF191024_1';
old_sess_date = '20191114';
old_sess = 'prunned_cells_2.mat';
new_sess_date = '20191210';
new_sess = 'prunned_cell_bodies.mat';
matched_rois = load('D:\Lukas\roi results\cellRegistered_LF191024_1_14_10_reg.mat');

roi_list = matched_rois.cell_registered_struct.cell_to_index_map;
true_pos = matched_rois.cell_registered_struct.true_positive_scores;
log_pos = isnan(true_pos);
log_pos = log_pos == 0;
left = roi_list(:,1);
left = left(log_pos);
right = roi_list(:,2);
right = right(log_pos);

A_left = load(['D:\Lukas\data\animals_raw\',mouse,'\',old_sess_date,'\',old_sess]);
%save_data1 = A_left.A_save;
save_data1 = A_left.a__append;
res_data_left = zeros(size(save_data1,1),size(save_data1,2),size(left,1));

for x = 1:size(left,1)
    num = left(x);
    res_data_left(:,:,x) = save_data1(:,:,num);
end


A_right = load(['D:\Lukas\data\animals_raw\',mouse,'\',new_sess_date,'\',new_sess]);
%save_data = A_right.A_save;
save_data = A_right.a_rev_a;
res_data_right = zeros(size(save_data,1),size(save_data,2),size(right,1));

for x = 1:size(right,1)
    num = right(x);
    res_data_right(:,:,x) = save_data(:,:,num);
end

res = zeros(size(left,1),size(left,2));

for x = 1:size(left)
figure;

tab1 = uitab('Title','Left');
ax1 = axes(tab1,'Position',[0.1 0.55 0.8 0.5]);
ax2 = axes(tab1,'Position',[0.1 0.05 0.8 0.5]);

%tab2 = uitab('Title','Right');
%ax2 = axes(tab2);

tab3 = uitab('Title','Both');
ax3 = axes(tab3);
set(gcf, 'Position',  [100, 100, 1100, 800])

image(ax1,res_data_left(:,:,x)*75);
image(ax2,res_data_right(:,:,x)*75);
left_right = res_data_left(:,:,x) + res_data_right(:,:,x);
image(ax3,left_right*75);

nnuumm = input('Match? (1 or 0) ');
res(x) = nnuumm;

close;
end

%%
resl = logical(res);
leftl = left(resl);
right1 = right(resl);
new_res = [leftl right1];

new_res_left = res_data_left(:,:,resl);
new_res_right = res_data_right(:,:,resl);
new_res_left = sum(new_res_left,3);
new_res_right = sum(new_res_right,3);
figure;
image((new_res_left+new_res_right)*75);
save(['D:\Lukas\data\animals_raw\',mouse,'\',new_sess_date,'\matched_cells.mat'],'new_res')


