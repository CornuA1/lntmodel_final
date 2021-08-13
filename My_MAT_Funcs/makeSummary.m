function makeSummary(fileExt,sessions,dff_files,dff_files_ol,things)


sessionExt = 'M01_000_00';
sessionExt2 = 'M01_000_01';
sessionEnd = '_results.mat';

for i=1:length(sessions)
    x = strcat(fileExt,int2str(sessions(i)),'\');
    if things(i) == 1
    y = strcat(sessionExt2,int2str(dff_files(i)),sessionEnd);
    else
    y = strcat(sessionExt,int2str(dff_files(i)),sessionEnd);
    end
    createHistogramFile(x,y);
end

for i=1:length(sessions)
    a = strcat(fileExt,int2str(sessions(i)),'_ol\');
    if things(i) == 1
    b = strcat(sessionExt2,int2str(dff_files_ol(i)),sessionEnd);
    else
    b = strcat(sessionExt,int2str(dff_files_ol(i)),sessionEnd);
    end
    createHistogramFile(a,b);
end


end