clear;

mouse = 'LF191023_blue';
old_sess_date = '20191204';
old_sess = 'prunned_cell_bodies.mat';
new_sess_date = '20191208';
new_sess = 'prunned_cell_bodies.mat';
one_sess_date = '20191210';
one_sess = 'prunned_cell_bodies.mat';
matched_rois = load('D:\Lukas\roi results\cellRegistered_LF191023_blue_04_08_10_reg.mat');

A_left = load(['D:\Lukas\data\animals_raw\',mouse,'\',old_sess_date,'\',old_sess]);
save_data1 = A_left.a_rev_a;

A_right = load(['D:\Lukas\data\animals_raw\',mouse,'\',new_sess_date,'\',new_sess]);
save_data = A_right.a__append;

A_right_r = load(['D:\Lukas\data\animals_raw\',mouse,'\',one_sess_date,'\',one_sess]);
save_datar = A_right_r.a__append;

figure;
image((sum(save_data1,3)+sum(save_data,3)+sum(save_datar,3))*50);