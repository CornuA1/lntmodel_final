function sbxPreBatch()
addpath('C:\Users\Lou\Documents\MATLAB\NoRMCorre-master','-end');
[sbxNames, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.', 'MultiSelect', 'on');
[~,width] = size(sbxNames);
for w=1:width
    sbxName = char(sbxNames(w));
    current_file = ['loading ',sbxName];
    disp(current_file);
    Info = sbxInfo([sbxPath, sbxName]);
    disp('sbx Process');
    sbxPreAnalysisNew(Info)
    disp('done');
end
end