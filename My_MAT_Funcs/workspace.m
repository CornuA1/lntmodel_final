clear;

rootdir = 'D:\Lukas\data\animals_raw\';
filelist = dir(fullfile(rootdir, '**\glm_fits_calcium_gaussian\*.mat'));
r2_vals = zeros(length(filelist),1);

for i=1:length(filelist)
    roi_loc = strcat(filelist(i).folder,'\',filelist(i).name);
    curR2 = load(roi_loc).r2_ave;
    r2_vals(i) = curR2;
end

[r2Order, I] = sort(r2_vals,'descend');
%%
for i=1:length(filelist)
    disp(filelist(I(i)));
    x = input('thoughts? ');
end