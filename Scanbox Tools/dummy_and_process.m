function dummy_and_process()
addpath('C:\Users\Lou\Documents\MATLAB\NoRMCorre-master','-end');
[sbxNames, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.', 'MultiSelect', 'on');
[~,width] = size(sbxNames);
for w=1:width
    sbxName = char(sbxNames(w));
    current_file = ['loading ',sbxName];
    disp(current_file);
    disp('norm corre');
    if w == 1
        disp('first');
        temp1 = norm_corre_process(sbxName, sbxPath);
    else
        disp('second');
        norm_corre_process(sbxName, sbxPath, temp1);
    end
    Info = sbxInfo([sbxPath, sbxName]);
    sbxName = char([Info.Directory.name,'_rigid.sbx']);
    Info = sbxInfo([sbxPath, sbxName]);
    disp('dummy alignment file');
    phaseDifferences = zeros(1,Info.maxIndex+1);
    rowShifts = zeros(1,Info.maxIndex+1);
    columnShifts = zeros(1,Info.maxIndex+1);
    frameCrop = [0,0,0,0];
    save([Info.Directory.folder, Info.Directory.name, '.rigid'], 'phaseDifferences', 'rowShifts', 'columnShifts', 'frameCrop');
    disp('sbx Process');
    sbxPreAnalysisNew(Info)
    disp('done');
end
end