clear;
%{
 x = load('Q:\Documents\Harnett UROP\LF191022_1\20191209\matched_cells.mat');
 match_list = x.new_res;
 y = load('Q:\Documents\Harnett UROP\LF191022_1\20191209\prunned_cell_bodies.mat');
 new_list = y.res_log;
 %}

rootdir = 'D:\Lukas\data\animals_raw\';
filelist = dir(fullfile(rootdir, '**\prunned_cell_bodies.mat'));
for i=1:length(filelist)
    name = filelist(i).folder;
 z = load(strcat(name,'\prunned_cell_bodies.mat'));
 old_list = z.res_log;
 trans = zeros(length(old_list),3);
 run_1 = 1;
 run_2 = 1;
 for x=1:length(old_list)
     trans(x,1) = run_1;
     run_1 = run_1 + 1;
     trans(x,2) = old_list(x,1);
     if old_list(x,1) == 1
         trans(x,3) = run_2;
         run_2 = run_2 + 1;
     end
 end
 save(strcat(name,'\cell_transfer_map.mat'),'trans')
end