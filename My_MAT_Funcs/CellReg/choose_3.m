clear;

mouse = 'LF191023_blue';
old_sess_date = '20191204';
old_sess = 'prunned_cell_bodies.mat';
new_sess_date = '20191208';
new_sess = 'prunned_cell_bodies.mat';
one_sess_date = '20191210';
one_sess = 'prunned_cell_bodies.mat';
matched_rois = load('D:\Lukas\roi results\cellRegistered_LF191023_blue_04_08_10_reg.mat');

roi_list = matched_rois.cell_registered_struct.cell_to_index_map;
log_roi_list = zeros(size(roi_list,1),1);
for x = 1:size(roi_list,1)
    if ~ismember(0,roi_list(x,:))
        log_roi_list(x) = 1;
    end
end
log_pos = logical(log_roi_list);
left = roi_list(:,1);
left = left(log_pos);
right = roi_list(:,2);
right = right(log_pos);
r_right = roi_list(:,3);
r_right = r_right(log_pos);

A_left = load(['D:\Lukas\data\animals_raw\',mouse,'\',old_sess_date,'\',old_sess]);
save_data1 = A_left.a_rev_a;
res_data_left = zeros(size(save_data1,1),size(save_data1,2),size(left,1));

for x = 1:size(left,1)
    num = left(x);
    res_data_left(:,:,x) = save_data1(:,:,num);
end

A_right = load(['D:\Lukas\data\animals_raw\',mouse,'\',new_sess_date,'\',new_sess]);
save_data = A_right.a__append;
res_data_right = zeros(size(save_data,1),size(save_data,2),size(right,1));

for x = 1:size(right,1)
    num = right(x);
    res_data_right(:,:,x) = save_data(:,:,num);
end

A_right_r = load(['D:\Lukas\data\animals_raw\',mouse,'\',one_sess_date,'\',one_sess]);
save_datar = A_right_r.a__append;
res_data_right_r = zeros(size(save_datar,1),size(save_datar,2),size(r_right,1));

for x = 1:size(r_right,1)
    num = r_right(x);
    res_data_right_r(:,:,x) = save_datar(:,:,num);
end

res = zeros(size(left,1),size(left,2));

for x = 1:size(left)
figure;

tab1 = uitab('Title','Left');
ax1 = axes(tab1,'Position',[0.1 0.55 0.8 0.5]);
ax2 = axes(tab1,'Position',[0.1 0.05 0.8 0.5]);
ax12 = axes(tab1,'Position',[0.1 0.05 0.8 0.5]);

%tab2 = uitab('Title','Right');
%ax2 = axes(tab2);

tab3 = uitab('Title','Both');
ax3 = axes(tab3);

image(ax1,res_data_left(:,:,x)*75);
image(ax2,res_data_right(:,:,x)*75);
image(ax12,res_data_right_r(:,:,x)*75);
left_right = res_data_left(:,:,x) + res_data_right(:,:,x) + res_data_right_r(:,:,x);
image(ax3,left_right*75);

nnuumm = input('Match? (1 or 0) ');
res(x) = nnuumm;

close;
end

%%
resl = logical(res);
leftl = left(resl);
right1 = right(resl);
r_rightl = r_right(resl);
new_res = [leftl right1 r_rightl];

new_res_left = res_data_left(:,:,resl);
new_res_right = res_data_right(:,:,resl);
new_res_right_r = res_data_right_r(:,:,resl);
new_res_left = sum(new_res_left,3);
new_res_right = sum(new_res_right,3);
new_res_right_r = sum(new_res_right_r,3);
figure;
image((new_res_left+new_res_right+new_res_right_r)*50);
saveas(gcf,['D:\Lukas\data\animals_raw\',mouse,'\',new_sess_date,'\matched_cells_pic_expert.png'])
nadf = input('Good?');
close;
save(['D:\Lukas\data\animals_raw\',mouse,'\',new_sess_date,'\matched_cells_expert.mat'],'new_res')


