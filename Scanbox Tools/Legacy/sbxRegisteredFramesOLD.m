function sbxRegisteredFrames(sbxName)
    
    %SBXREGISTEREDFRAMES Read, motion-correct, and save imaging data from an .sbx file.
    %   SBXREGISTEREDFRAMES(sbxName) outputs motion-corrected .png files from the selected .sbx file.
    %
    %   sbxName: string 
    %       Path of .sbx file to be analyzed (e.g., 'C:/User/xx0_000_001.sbx').
    
    % specify cropping in pixels: [from left, from right, from top, from bottom]
    %frameCrop = [95, 0, 5, 0]; % somas
    %frameCrop = [95, 0, 5, 5]; % dendrites
    frameCrop = [0, 0, 0, 0];
    
    % set number of pixels to correct resonance offset for bidirectional scanning
    resonanceOffset = 0;
    
    % indicate if we want to skip over any frames (defaults are 0 and 1)
    frameOffset = 0;
    frameStep = 1;
    
    % set the number of consecutive times to motion correct
    passes = 2;
    
    % set number of (random) frames to use in generating a reference image
    nFrames = 1000;
    
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
    
    if ~isdir([sbxPath, sbxName])
        mkdir([sbxPath, sbxName])
    end
    
    imwrite(reference, [sbxPath, sbxName, '_reference', '.png'], 'bitdepth', 16);
    
    subpixelFactor = 1;
    
    disp('Saving registered images...');
    
    for f = 1:length(frameIndices)
        if f == uint32(length(frameIndices)/2)
            disp('50% complete...');
        end
        
        frame = sbxRead(Info, frameIndices(f));
        
        % apply resonance offset to every other line
        for j = 2:2:size(frame, 1)
            frame(j, 1:end) = [frame(j, 1 + resonanceOffset:end), zeros(1, resonanceOffset)];
        end
        
        % crop as indicated
        frame = frame(frameCrop(3) + 1:Info.sz(1) - frameCrop(4), frameCrop(1) + 1:Info.sz(2) - frameCrop(2));
        
        [~, registeredFrame] = dftregistration(fft2(reference), fft2(frame), subpixelFactor);

        % scale the image pixel values up so that they are meaningful for .png
        registeredFrame = abs(ifft2(registeredFrame));
        
        imwrite(uint16((registeredFrame*max(max(double(frame))))/max(max(registeredFrame))), [sbxPath, sbxName, '\registered_', int2str(f), '.png'], 'bitdepth', 16);
    end
    
    disp('Done.');
    
    if passes > 1
        for pass = 2:passes
            reference = zeros(Info.sz(1) - frameCrop(3) - frameCrop(4), Info.sz(2) - frameCrop(1) - frameCrop(2));
            indices = randperm(Info.maxIndex + 1, sampleSize) - 1;
            
            disp(['Pass ', int2str(pass)]);
            disp('Generating a reference image...');

            for f = 1:nFrames
                frame = imread([sbxPath, sbxName, '\registered_', int2str(indices(f)), '.png']);

                % convert to double so we don't saturate the uint16 precision
                reference = reference + double(frame);
            end

            reference = reference/nFrames;
            reference = uint16(reference);

            imwrite(reference, [sbxPath, sbxName, '_reference', '.png'], 'bitdepth', 16);

            disp('Saving registered images...');

            for f = 1:length(frameIndices)
                if f == uint32(length(frameIndices)/2)
                    disp('50% complete...');
                end

                frame = imread([sbxPath, sbxName, '\registered_', int2str(f), '.png']);

                [~, registeredFrame] = dftregistration(fft2(reference), fft2(frame), subpixelFactor);

                % scale the image pixel values up so that they are meaningful for .png
                registeredFrame = abs(ifft2(registeredFrame));

                imwrite(uint16((registeredFrame*max(max(double(frame))))/max(max(registeredFrame))), [sbxPath, sbxName, '\registered_', int2str(f), '.png'], 'bitdepth', 16);
            end
            
            disp('Done.');
        end
    end
   
end