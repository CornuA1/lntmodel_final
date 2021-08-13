function sbxRegister(sbxName)

    %SBXREGISTER Motion-correct imaging data contained in an .sbx file.
    %   SBXREGISTER(sbxName) registers imaging data frame-by-frame and saves to a new .sbx file.
    %
    %   sbxName: string 
    %       Path of .sbx file to be analyzed (e.g., 'C:/User/xx0_000_001.sbx').
    
    % specify cropping in pixels: [from left, from right, from top, from bottom]
    %frameCrop = [95, 0, 35, 0]; % somas
    %frameCrop = [95, 0, 125, 5]; % dendrites
    %frameCrop = [95, 0, 0, 0]; % bidirectional correction
    %frameCrop = [95, 0, 10, 0]; % lukas
    frameCrop = [0, 0, 0, 0];
    
    % set number of pixels to correct resonance offset for bidirectional scanning
    resonanceOffset = 0;
    
    % indicate if we want to skip over any frames (defaults are 0 and 1)
    frameOffset = 0;
    frameStep = 1;
    
    % set the number of consecutive times to motion correct
    passes = 3;
    
    % set number of (random) frames to use in generating a reference image
    sampleSize = 1000;
    
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

    if isfield(Info, 'registered') && Info.registered == 1
        waitfor(msgbox('Error: Data has already been registered.'));
        error('Data has already been registered.');
    end
    
    frameIndices = frameOffset:frameStep:Info.maxIndex;
    
    reference = zeros(Info.sz);
    
    indices = randperm(Info.maxIndex + 1, sampleSize) - 1;
    
    disp('Pass 1');
    disp('Generating a reference image...');
    
    for f = 1:sampleSize
        
        frame = sbxRead(Info, indices(f));
        
        % convert to double so we don't saturate the uint16 precision
        reference = reference + double(frame);
    end
    
    reference = reference/sampleSize;
    reference = uint16(reference);
    
    % apply resonance offset to every other line
    if resonanceOffset ~= 0
        for j = 2:2:size(reference, 1)
            reference(j, 1:end) = [reference(j, 1 + resonanceOffset:end), zeros(1, resonanceOffset)];
        end
    end

    % crop as indicated
    reference = reference(frameCrop(3) + 1:Info.sz(1) - frameCrop(4), frameCrop(1) + 1:Info.sz(2) - frameCrop(2));
    
    subpixelFactor = 1;
    
    % get the number of bytes to assign to the new file, taking into account cropping and uint16 precision
    samplesPerFrame = (Info.sz(1) - frameCrop(3) - frameCrop(4))*(Info.sz(2) - frameCrop(1) - frameCrop(2))*Info.nChannels;
    bytesPerFrame = 2*samplesPerFrame;
    nBytes = bytesPerFrame*(length(frameIndices) + 1);

    registeredSBXName = [sbxPath, sbxName, '_registered'];
    [~, ~] = system(sprintf('fsutil file createnew %s %d', [registeredSBXName, '.sbx'], nBytes));
    fileID = fopen([registeredSBXName, '.sbx'], 'w');
    
    % create new .mat file
    info = importdata([sbxPath, sbxName, '.mat']);
    
    info.resonanceOffset = resonanceOffset;
    if isfield(info, 'timeStamps')
        info.timeStamps = info.timeStamps(frameOffset + 1:frameStep:end);
    end
    
    % indicate that .sbx file has been motion-corrected
    info.registered = 1;
    
    % indicate changes to frames
    info.sz = [info.sz(1) - frameCrop(3) - frameCrop(4), info.sz(2) - frameCrop(1) - frameCrop(2)];

    save([registeredSBXName, '.mat'], 'info');
    
    disp('Registering imaging data...');
    
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
        
        [~, registeredFrame] = dftregistration(fft2(reference), fft2(frame), subpixelFactor);

        % reshape the data into its original, pre-processed form
        registeredFrame = abs(ifft2(registeredFrame));
        registeredFrame = uint16((registeredFrame*max(max(double(frame))))/max(max(registeredFrame)));
        
        if Info.nChannels > 1
            registeredFrame = permute(intmax('uint16') - registeredFrame, [1, 3, 2]);
        else
            registeredFrame = permute(intmax('uint16') - registeredFrame, [2, 1]);
        end
        
        registeredFrame = reshape(registeredFrame, [samplesPerFrame, 1]);
        
        fseek(fileID, f*Info.bytesPerFrame, 'bof');
        fwrite(fileID, registeredFrame, 'uint16');
    end
    
    disp('Done.');
    
    if passes > 1
        for pass = 2:passes
            reference = zeros(info.sz(1), info.sz(2));
            indices = randperm(Info.maxIndex + 1, sampleSize) - 1;
            
            Info = sbxInfo(registeredSBXName);
            
            disp(['Pass ', int2str(pass)]);
            disp('Generating a reference image...');

            for f = 1:sampleSize
                frame = sbxRead(Info, indices(f));
        
                % convert to double so we don't saturate the uint16 precision
                reference = reference + double(frame);
            end

            reference = reference/sampleSize;
            reference = uint16(reference);

            disp('Registering imaging data...');

            for f = 0:Info.maxIndex
                if f == uint32((Info.maxIndex + 1)/2)
                    disp('50% complete...');
                end

                frame = sbxRead(Info, f);

                [~, registeredFrame] = dftregistration(fft2(reference), fft2(frame), subpixelFactor);

                % reshape the data into its original, pre-processed form
                registeredFrame = abs(ifft2(registeredFrame));
                registeredFrame = uint16((registeredFrame*max(max(double(frame))))/max(max(registeredFrame)));

                if Info.nChannels > 1
                    registeredFrame = permute(intmax('uint16') - registeredFrame, [1, 3, 2]);
                else
                    registeredFrame = permute(intmax('uint16') - registeredFrame, [2, 1]);
                end

                registeredFrame = reshape(registeredFrame, [Info.samplesPerFrame, 1]);
        
                fseek(fileID, f*Info.bytesPerFrame, 'bof');
                fwrite(fileID, registeredFrame, 'uint16');
            end
            
            disp('Done.');
        end
    end
        
    fclose(fileID);

end