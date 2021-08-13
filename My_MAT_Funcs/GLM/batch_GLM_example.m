clear;
%{
fileExt = 'D:\Lukas\data\animals_raw\LF191023_blue\2019';
sessions = [1113,1114,1119,1121,1125,1204,1206,1208,1210,1212,1215,1217];
dff_files = [3,1,4,0,1,0,2,2,6,2,3,6];
dff_files_ol = [4,2,5,1,2,1,3,3,7,3,4,7];
things = [0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0];
makeMouseBatch(fileExt,sessions,dff_files,dff_files_ol,things)
%}

fileExt = 'D:\Lukas\data\animals_raw\LF191024_1\2019';
sessions = [1114,1115,1121,1204,1207,1210];
dff_files = [0,1,0,1,4,2];
dff_files_ol = [1,2,1,2,5,3];
things = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0];
makeMouseBatch(fileExt,sessions,dff_files,dff_files_ol,things)