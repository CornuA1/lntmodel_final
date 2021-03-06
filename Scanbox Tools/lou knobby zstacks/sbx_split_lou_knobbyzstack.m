% function result = sbx_split_lou_knobbyzstack(Info, Parameters)

    %SBXSPLIT Split up volumetric scanning data into single planes.
    %   result = SBXSPLIT(Info, Parameters) splits volumetric scanning data into single .sbx and .mat files per imaging plane.
    %
    %   Info: structure 
    %       Info structure generated by sbxInfo from corresponding .mat file.
    %
    %   Parameters: structure
    %       Optional input containing parameter specifications.
    %
    %   result: string
    %       Used by sbxAnalysis to confirm if function ran correctly or was cancelled.
    
%     if ~exist('Info', 'var')
        try
            [sbxName, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.');
        catch
            waitfor(msgbox('Error: Please select valid .sbx file.'));
            error('Please select valid .sbx file.');
        end
    
        % pull off the file extension
        sbxName = strtok(sbxName, '.');

        Info = sbxInfo([sbxPath, sbxName]);
%     end
  

knobstuff=Info.config.knobby.schedule;

%%
scanningOrder =0:5:sum(knobstuff(:,3))-5;
nScans = length(scanningOrder);
planes=scanningOrder;
nPlanes = length(planes)
framesperplane=knobstuff(1,end);



    % get rid of the first period - there are always artefacts there
    volumetricOffset = 0;
    
    for s = 1:nScans
        for p = 1:nPlanes
            if scanningOrder(s) == planes(p)
                scanningOrder(s) = p;
                break
            end
        end
    end

    newFileIDs = zeros(1, nPlanes);
    fileNames = cell(1, nPlanes);
    
 
    % create new .sbx and .mat files
    for p = 1:nPlanes
        fileNames{p} = [Info.Directory.folder, Info.Directory.name, '_', sprintf('%04d', planes(p))];
        newFileIDs(p) = fopen([fileNames{p}, '.sbx'], 'w');
        
        info = importdata([Info.Directory.folder, Info.Directory.name, '.mat']);
        
        % indicate that .sbx file has been split and how it was done
        info.split = true;
        info.plane = [p, planes(p)];
        info.scanningOrder = scanningOrder;  
        info.volumetricOffset = volumetricOffset;
        
        if isfield(info, 'timeStamps')
            info.timeStamps = Info.timeStamps(volumetricOffset + p:nPlanes:end);
        end
        
        save([fileNames{p}, '.mat'], 'info');
%         
%         if GUI
%             if getappdata(progressBar, 'Canceling')
%                 delete(progressBar);
%                 result = 'Canceled';
%                 return
%             else
%                 waitbar(p/nPlanes, progressBar);
%             end
%         end
    end
    
    

    oldFileID = fopen([Info.Directory.folder, Info.Directory.name, '.sbx'], 'r');
    
    
       %%
       if Info.maxIndex>length(planes)*framesperplane-1
           Info.maxIndex=length(planes)*framesperplane-1
       end
       
       
    % split file
    for i = 0:Info.maxIndex - volumetricOffset        
        fseek(oldFileID, (i + volumetricOffset)*Info.bytesPerFrame, 'bof');
        frame = fread(oldFileID, Info.samplesPerFrame, 'uint16=>uint16');
        fwrite(newFileIDs(scanningOrder(floor([i]/framesperplane)+1)), frame, 'uint16');
        
        
    end

    
    
    %%
    for p = 1:nPlanes
        fclose(newFileIDs(p));
    end
    
    fclose(oldFileID);
%     
%     if GUI
%         delete(progressBar);
%     end
    
%     result = 'Completed';

% end