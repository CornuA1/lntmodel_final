clear all
close all
clc

index=[]
for kk=1:150
    index=[index;'000'];
end

for kk=2
    index(kk,3)=num2str(5*(kk-1))
end

for kk=3:20
    index(kk,2:3)=num2str(5*(kk-1))
end

for kk=21:150
    index(kk,1:3)=num2str(5*(kk-1))
end

%%
for kk=1:150

    idx=index(kk,:)
    if kk<121
        stacknumbers=['2'; '3';'4';'5';'6';'7']
    else
        stacknumbers=['2'; '3';'4';'5';'6']
    end
    

sbxName=['pizzaman4_000_00',stacknumbers(1),'_0',idx,'.sbx']
sbxPath='F:\Quique\pizzaman stack\pizzaman4\'

% pull off the file extension
% sbxName = strtok(sbxName, '.');

Info = sbxInfo([sbxPath, sbxName]);


fileNames{1} = [Info.Directory.folder, idx, 'compiledstack'];
newFileIDs(1) = fopen([fileNames{1}, '.sbx'], 'w');
info = importdata([Info.Directory.folder, Info.Directory.name, '.mat'])


oldFileID = fopen([Info.Directory.folder, Info.Directory.name, '.sbx'], 'r');

% write in new file
for i = 0:Info.maxIndex
    fseek(oldFileID, (i)*Info.bytesPerFrame, 'bof');
    frame = fread(oldFileID, Info.samplesPerFrame, 'uint16=>uint16');
    fwrite(newFileIDs(1), frame, 'uint16');
end


totalframe=Info.maxIndex;


for kk=2:length(stacknumbers)
    sbxName=['pizzaman4_000_00',stacknumbers(kk),'_0',idx,'.sbx']
        
    % pull off the file extension
    sbxName = strtok(sbxName, '.');
    
    Info = sbxInfo([sbxPath, sbxName]);
    info = importdata([Info.Directory.folder, Info.Directory.name, '.mat'])
    oldFileID = fopen([Info.Directory.folder, Info.Directory.name, '.sbx'], 'r');
    
    % add to new file
    for i = 0:Info.maxIndex
        fseek(oldFileID, (i )*Info.bytesPerFrame, 'bof');
        frame = fread(oldFileID, Info.samplesPerFrame, 'uint16=>uint16');
        fwrite(newFileIDs(1), frame, 'uint16');
         
    end
    totalframe=totalframe+Info.maxIndex;
    
end


info.maxIndex=totalframe
save([fileNames{1}, '.mat'], 'info');

fclose(newFileIDs(1));


fclose(oldFileID);

fclose('all')
clearvars -except index
end