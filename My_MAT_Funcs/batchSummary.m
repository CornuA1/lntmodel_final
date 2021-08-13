clear;

fileExt = 'D:\Lukas\data\animals_raw\LF191022_1\2019';
sessions = [1213,1215,1217]; % 1114,1115,1121,1125,1204,1207,1209,1211,
dff_files = [4,5,0]; %2,4,3,3,2,0,0,0,
dff_files_ol = [5,6,1]; %3,6,4,4,3,1,1,1,
things = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
makeMouseBatch(fileExt,sessions,dff_files,dff_files_ol,things)

fileExt = 'D:\Lukas\data\animals_raw\LF191022_2\2019';
sessions = [1114,1116,1121,1204,1206,1208,1210,1212,1216];
dff_files = [4,0,5,4,4,4,8,4,0];
dff_files_ol = [5,1,6,7,5,5,9,5,1];
things = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
makeMouseBatch(fileExt,sessions,dff_files,dff_files_ol,things)

fileExt = 'D:\Lukas\data\animals_raw\LF191022_3\2019';
sessions = [1113,1114,1119,1121,1125,1204,1207,1210,1211,1215,1217];
dff_files = [0,0,0,3,7,8,2,0,2,1,2];
dff_files_ol = [1,1,1,4,8,9,3,1,3,2,3];
things = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0];
makeMouseBatch(fileExt,sessions,dff_files,dff_files_ol,things)

fileExt = 'D:\Lukas\data\animals_raw\LF191023_blank\2019';
sessions = [1114,1116,1121,1206,1208,1210,1212,1213,1216,1217];
dff_files = [6,2,8,0,0,4,0,2,3,4];
dff_files_ol = [7,3,9,1,1,5,1,3,4,5];
things = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
makeMouseBatch(fileExt,sessions,dff_files,dff_files_ol,things)

fileExt = 'D:\Lukas\data\animals_raw\LF191023_blue\2019';
sessions = [1113,1114,1119,1121,1125,1204,1206,1208,1210,1212,1215,1217];
dff_files = [3,1,4,0,1,0,2,2,6,2,3,6];
dff_files_ol = [4,2,5,1,2,1,3,3,7,3,4,7];
things = [0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0];
makeMouseBatch(fileExt,sessions,dff_files,dff_files_ol,things)

fileExt = 'D:\Lukas\data\animals_raw\LF191024_1\2019';
sessions = [1114,1115,1121,1204,1207,1210];
dff_files = [0,1,0,1,4,2];
dff_files_ol = [1,2,1,2,5,3];
things = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0];
makeMouseBatch(fileExt,sessions,dff_files,dff_files_ol,things)


disp('Donezo!');

