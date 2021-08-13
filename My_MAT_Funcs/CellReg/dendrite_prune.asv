clear;

%{
Description: 
    Hand prunes ROIs of dendrites and junk identified during
    caiman.



%}
%%
mouse = 'LF191022_1';
old_sess_date = '20191204';
old_sess = 'M01_000_002_results.mat';

A = load(['D:\Lukas\data\animals_raw\',mouse,'\',old_sess_date,'\',old_sess]);
save_data = A.A_save;
res_data = zeros(size(save_data,3),1);

for x = 1:size(save_data,3)
figure;
image(save_data(:,:,x)*180);
set(gcf, 'Position',  [100, 100, 1692, 1124])
nnuumm = input('Match? (1 or 0) ');
while size(nnuumm) == size([])
    nnuumm = input('Match? (1 or 0) ');
end
res_data(x) = nnuumm;
close;
end

res_log = logical(res_data);
a_rev_a = save_data(:,:,res_log);
new_rev_res = sum(a_rev_a,3);
figure;
%{
image(new_rev_res*75);
saveas(gcf,['D:\Lukas\data\animals_raw\',mouse,'\',old_sess_date,'\prunned_cells.png']);
throw = input('Good?');
close;
save(['D:\Lukas\data\animals_raw\',mouse,'\',old_sess_date,'\prunned_cell_bodies.mat'],'a_rev_a','res_log')
disp(strcat('Done with---',mouse,'---and---',old_sess_date));
%}