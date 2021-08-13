function sbxSplit(sbxName)

    %SBXSPLIT Split up volumetric scanning data into single planes.
    %   SBXSPLIT(sbxName) splits volumetric scanning data into single .sbx and .mat files per imaging plane.
    %
    %   sbxName: string 
    %       Path of .sbx file to be analyzed (e.g., 'C:/User/xx0_000_001.sbx').
    
    % set the frame offset for when actual volumetric scanning began; adjust this till the first frame corresponds to the first scan of a full scanning waveform
    volumetricOffset = 2;
    
    if ~exist('sbxName', 'var')
        try
            [sbxName, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.');
        catch
            waitfor(msgbox('Error: Please select valid .sbx file.'));
            error('Please select valid .sbx file.');
        end
    end
    
    % pull off the file extension
    sbxName = strtok(sbxName, '.');
            
    Info = sbxInfo([sbxPath, sbxName]);

    if isfield(Info, 'split') && Info.split == 1
        waitfor(msgbox('Error: Data has already been split.'));
        error('Data has already been split.');
    end
        
    if Info.volscan == 0
        waitfor(msgbox('Error: Data does not contain volumetric scans.'));
        error('Data does not contain volumetric scans.');
    end
    
    % get the number and order of scanning planes
    scanningOrder = Info.otwave_um;
    nScans = length(scanningOrder);
    planes = unique(scanningOrder);
    nPlanes = length(planes);
    
    for s = 1:nScans
        for p = 1:nPlanes
            if scanningOrder(s) == planes(p)
                scanningOrder(s) = p;
                break
            end
        end
    end
    
    bytesPerSlice = Info.Directory.bytes/nPlanes;

    newFileIDs = zeros(1, nPlanes);
    fileNames = cell(1, nPlanes);

    % create new .sbx files
    for p = 1:nPlanes
        fileNames{p} = [sbxPath, sbxName, '_', sprintf('%04d', planes(p))];
        [~, ~] = system(sprintf('fsutil file createnew %s %d', [fileNames{p}, '.sbx'], bytesPerSlice));
        newFileIDs(p) = fopen([fileNames{p}, '.sbx'], 'w');
    end

    % create new .mat files
    for p = 1:nPlanes
        info = importdata([sbxPath, sbxName, '.mat']);
        
        % indicate that .sbx file has been split and how it was done
        info.split = 1;
        info.plane = [p, planes(p)];
        info.scanningOrder = scanningOrder;  
        info.volumetricOffset = volumetricOffset;
        
        if isfield(info, 'timeStamps')
            info.timeStamps = Info.timeStamps(volumetricOffset + p - 1:nPlanes:end);
        end
        
        save([fileNames{p}, '.mat'], 'info');
    end
    
    oldFileID = fopen([sbxPath, sbxName, '.sbx'], 'r');
    
    disp('Splitting volumetric scans...');
    
    % split file
    for i = 0:Info.maxIndex - volumetricOffset
        if i == uint32((Info.maxIndex - volumetricOffset)/2)
            disp('50% complete...');
        end
        
        fseek(oldFileID, (i + volumetricOffset)*Info.bytesPerFrame, 'bof');
        frame = fread(oldFileID, Info.samplesPerFrame, 'uint16=>uint16');
        fwrite(newFileIDs(scanningOrder(mod(i, nScans) + 1)), frame, 'uint16');
    end
    
    disp('Done.');

    for p = 1:nPlanes
        fclose(newFileIDs(p));
    end
    
    fclose(oldFileID);

end