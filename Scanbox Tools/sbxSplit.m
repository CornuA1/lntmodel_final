function result = sbxSplit(Info, Parameters)

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
    
    if ~exist('Info', 'var')
        try
            [sbxName, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.');
        catch
            waitfor(msgbox('Error: Please select valid .sbx file.'));
            error('Please select valid .sbx file.');
        end
    
        % pull off the file extension
        sbxName = strtok(sbxName, '.');

        Info = sbxInfo([sbxPath, sbxName]);
    end
        
    if Info.volscan == 0
        waitfor(msgbox('Error: Data does not contain volumetric scans.'));
        error('Data does not contain volumetric scans.');
    end

    if isfield(Info, 'split') && Info.split
        waitfor(msgbox('Error: Data has already been split.'));
        error('Data has already been split.');
    end
    
    % set the frame offset for when actual volumetric scanning began; adjust this till the first frame corresponds to the first scan of a full scanning waveform
    if ~exist('Parameters', 'var')
        GUI = false;
    else
        if ~isfield(Parameters, 'GUI')
            GUI = false;
        else
            GUI = Parameters.GUI;
        end
    end
    
    % get the number and order of scanning planes
    if ~isempty(Info.otwave_um)
        scanningOrder = Info.otwave_um;
    else
        scanningOrder = Info.otwave;
    end
    
    nScans = length(scanningOrder);
    planes = unique(scanningOrder);
    nPlanes = length(planes);
    
    % get rid of the first period - there are always artefacts there
    volumetricOffset = nScans;
    
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
    
    if GUI
        progressBar = waitbar(0, 'Creating new .sbx files...', 'Name', [Info.Directory.name, ': sbxSplit'], 'CreateCancelBtn', 'setappdata(gcbf, ''Canceling'', 1)');
        setappdata(progressBar, 'Canceling', 0);
    end
        
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
        
        if GUI
            if getappdata(progressBar, 'Canceling')
                delete(progressBar);
                result = 'Canceled';
                return
            else
                waitbar(p/nPlanes, progressBar);
            end
        end
    end
    
    oldFileID = fopen([Info.Directory.folder, Info.Directory.name, '.sbx'], 'r');
    
    if GUI
        if getappdata(progressBar, 'Canceling')
            delete(progressBar);
            result = 'Canceled';
            return
        else
            waitbar(0, progressBar, 'Splitting volumetric scans...');
        end
    end
    
    % split file
    for i = 0:Info.maxIndex - volumetricOffset        
        fseek(oldFileID, (i + volumetricOffset)*Info.bytesPerFrame, 'bof');
        frame = fread(oldFileID, Info.samplesPerFrame, 'uint16=>uint16');
        fwrite(newFileIDs(scanningOrder(mod(i + volumetricOffset, nScans) + 1)), frame, 'uint16');
        
        if GUI
            if getappdata(progressBar, 'Canceling')
                delete(progressBar);
                result = 'Canceled';
                return
            else
                waitbar((i + 1)/(Info.maxIndex - volumetricOffset + 1), progressBar);
            end
        end
    end

    for p = 1:nPlanes
        fclose(newFileIDs(p));
    end
    
    fclose(oldFileID);
    
    if GUI
        delete(progressBar);
    end
    
    result = 'Completed';

end