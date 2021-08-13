clear;
%{
naive = [('LF191022_1','20191115'),('LF191022_3','20191113'),('LF191023_blue','20191119'),('LF191022_2','20191116'),('LF191023_blank','20191114'),('LF191024_1','20191114')]
expert = [('LF191022_1','20191209'),('LF191022_3','20191207'),('LF191023_blue','20191208'),('LF191022_2','20191210'),('LF191023_blank','20191210'),('LF191024_1','20191210')]
%}
mouse = 'LF191022_1';
session1 = '20191115';
session2 = '20191209';
 
naive = load(['D:\Lukas\data\animals_raw\',mouse,'\',session1,'\prunned_cell_bodies.mat']);
expert = load(['D:\Lukas\data\animals_raw\',mouse,'\',session2,'\prunned_cell_bodies.mat']);
least = 10000000000000;
xx = -50:50;
yy = -50:50;

for i=1:length(xx)
    x = xx(i);
    for j=1:length(yy)
        y = yy(i);
        new_bodies = circshift(circshift(naive.a_rev_a,x,1),y,2);
        total = sum(expert.a_rev_a,3)+sum(new_bodies,3);
        zero_elms = length(find(~total));
        if zero_elms < least
            least = zero_elms;
            x_y = [x,y];
        end
    end
end

best_Cell_bodies = circshift(circshift(naive.a_rev_a,x_y(1),1),x_y(2),2);

figure;
image((sum(expert.a_rev_a,3)+sum(best_Cell_bodies,3))*120)

res_log = naive.res_log;
%save(['D:\Lukas\data\animals_raw\',mouse,'\',session,'_ol\adjusted_cell_bodies.mat'],'new_bodies','res_log')



%{
figure;
image((sum(expert.a_rev_a,3))*100)
figure;
image((sum(new_bodies,3))*100)
%}