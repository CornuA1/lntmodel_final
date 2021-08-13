clear;

sessionExt = 'M01_000_00';
sessionExt2 = 'M01_000_01';
sessionEnd = '_results.mat';
fileExt = 'D:\Lukas\data\animals_raw\LF191024_1\2019';
sessions = [1114,1115,1121,1204,1207,1210];
dff_files = [0,1,0,1,4,2];
dff_files_ol = [1,2,1,2,5,3];
things = [0,0,0,1,0,0,0,0,0,0,0];

for i=1:length(sessions)
    x = strcat(fileExt,int2str(sessions(i)),'\');
    if things(i) == 1
    y = strcat(sessionExt2,int2str(dff_files(i)),sessionEnd);
    else
    y = strcat(sessionExt,int2str(dff_files(i)),sessionEnd);
    end
    createGLMdata(x,y,'calcium','gaussian');
end

for i=1:length(sessions)
    a = strcat(fileExt,int2str(sessions(i)),'_ol\');
    if things(i) == 1
    b = strcat(sessionExt2,int2str(dff_files_ol(i)),sessionEnd);
    else
    b = strcat(sessionExt,int2str(dff_files_ol(i)),sessionEnd);
    end
    create_glm_files(a,b,'calcium','gaussian');
    createGLMdata(a,b,'calcium','gaussian');
end

disp('Donezo!');