clear;

load('Q:\Documents\Harnett UROP\LF191024_1\20191210\matched_cells.mat');
load('Q:\Documents\Harnett UROP\LF191024_1\20191210\cell_transfer_map.mat');

res = zeros(length(new_res),1);
for x=1:length(new_res); res(x) = find(trans(:,3) == new_res(x,2)); end
rep = sort(res);
%{
loc_info = 'Q:\Documents\Harnett UROP\';
MOUSE = 'LF191022_1\';
SESSION = '20191209\';

vall = false;
for i = 1:length(rep)
    save_nm = strcat('coef_',int2str(rep(i)));
    file_roi = strcat(loc_info, MOUSE, SESSION, 'data_glm_fit\',int2str(rep(i)),'.mat');
    data_roit = load(file_roi);
    coef_short = data_roit.coef_short(2:end);
    r2_short = data_roit.r2_short;
    coef_long = data_roit.coef_long(2:end);
    r2_long = data_roit.r2_long;
    if ~vall
        status = mkdir(strcat(loc_info,MOUSE,SESSION,'histo_files\'));
        vall = true;
    end
    save(strcat(loc_info,MOUSE,SESSION,'histo_files\',save_nm,'.mat'),'coef_short','coef_long','r2_short','r2_long')
end
%}