function sbxFrames(sbxName)
    
    %SBXFRAMES Read and save frames from an .sbx file.
    %   SBXFRAMES(sbxName) outputs .png files from the selected .sbx file.
    %
    %   sbxName: string 
    %       Path of .sbx file to be analyzed (e.g., 'C:/User/xx0_000_001.sbx').
    
    % specify cropping in pixels: [from left, from right, from top, from bottom]
    %frameCrop = [95, 75, 35, 0]; % somas
    %frameCrop = [95, 0, 125, 5]; % dendrites
    %frameCrop = [95, 75, 0, 0]; % bidirectional correction
    %frameCrop = [95, 0, 10, 0]; % lukas
    frameCrop = [0, 0, 0, 0];
    
    % set number of pixels to correct resonance offset for bidirectional scanning
    resonanceOffset = 0;
    
    % indicate if we want to skip over any frames (defaults are 0 and 1)
    frameOffset = 0;
    frameStep = 1;
    
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
    
    frameIndices = frameOffset:frameStep:Info.maxIndex;
    
    if ~isdir([sbxPath, sbxName])
        mkdir([sbxPath, sbxName])
    end
    
    disp('Saving imaging frames...');
    
    for f = 1:length(frameIndices)
        if f == uint32(length(frameIndices)/2)
            disp('50% complete...');
        end
        
        frame = sbxRead(Info, frameIndices(f));
        
        % apply resonance offset to every other line
        if resonanceOffset ~= 0
            for j = 2:2:size(frame, 1)
                frame(j, 1:end) = [frame(j, 1 + resonanceOffset:end), zeros(1, resonanceOffset)];
            end
        end
        
        % crop as indicated
        frame = frame(frameCrop(3) + 1:Info.sz(1) - frameCrop(4), frameCrop(1) + 1:Info.sz(2) - frameCrop(2));
        
        imwrite(frame, [sbxPath, sbxName, '\', int2str(f), '.png'], 'bitdepth', 16);
    end
    
    % reassure user
    disp('Done.');
    
end