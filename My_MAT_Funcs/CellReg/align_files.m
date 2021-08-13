clear;

mouse = 'LF191024_1';
old_sess_date = '20191210';
old_sess = 'prunned_cell_bodies.mat';
new_sess_date = '20191114';
new_sess = 'prunned_cell_bodies.mat';
%new_sess_date = '20191217';
%new_sess = 'prunned_cell_bodies.mat';

A_left = load(['D:\Lukas\data\animals_raw\',mouse,'\',old_sess_date,'\',old_sess]);
%save_data1 = A_left.a_rev_a;
save_data1 = A_left.a_rev_a;
res_data_left = zeros(size(save_data1,1),size(save_data1,2));

for x = 1:size(save_data1,3)
    res_data_left = res_data_left + save_data1(:,:,x);
end

A_right = load(['D:\Lukas\data\animals_raw\',mouse,'\',new_sess_date,'\',new_sess]);
save_data = A_right.a_rev_a;
res_data_right = zeros(size(save_data,1),size(save_data,2));

for x = 1:size(save_data,3)
    res_data_right = res_data_right + save_data(:,:,x);
end

thing = [-100,-50];
a__append = circshift(save_data,[-100,-50,0]);
res_log = A_right.res_log;
save(['D:\Lukas\data\animals_raw\',mouse,'\',new_sess_date,'\','prunned_cells_2.mat'],'a__append','res_log');
figure;
image(res_data_left*75);
figure;
image(circshift(res_data_right,thing)*75);
figure;
image((res_data_left+circshift(res_data_right,thing))*75);
input('yo');
close all;