function     [meanImage, maxIntensityImage, primage50, primage75, primage90,result] = sbxPreAnalysis_lou2(Info, Parameters)

  

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

    if exist([Info.Directory.folder, Info.Directory.name, '.pre'], 'file')
        waitfor(msgbox('Error: Data has already been pre-computed.'));
        error('Data has already been pre-computed.');
    end
    
    if ~exist('Parameters', 'var')
        GUI = false;
        
        resonantOffset = 0;
        
        % set the pixel intensity value to use as a threshold for selecting reference frames
        threshold = 0;
        
        % spatial smoothing at this step sometimes helps, but will generally make the cross-correlation image less helpful by suppressing non-correlated activity
        gaussianFilter = 0.0;
    
        % indicate cross-correlation method
        method = 1;

        % set sample size for mean image
        sampleSize = 500;
        
        % set min and max frame index to include in stack
        sampleBounds = [0, Info.maxIndex];
        
        % set quantile to use for max intensity projection
        maxQuantile = 0.95;
    else
        if ~isfield(Parameters, 'GUI')
            GUI = false;
        else
            GUI = Parameters.GUI;
        end
        if ~isfield(Parameters, 'resonantOffset')
            resonantOffset = 0;
        else
            resonantOffset = Parameters.resonantOffset;
        end
        if ~isfield(Parameters, 'threshold')
            threshold = 0;
        else
            threshold = Parameters.threshold;
        end
        if ~isfield(Parameters, 'gaussianFilter')
            gaussianFilter = 0.0;
        else
            gaussianFilter = Parameters.gaussianFilter;
        end
        if ~isfield(Parameters, 'method')
            method = 1;
        else
            method = Parameters.method;
        end
        if ~isfield(Parameters, 'sampleSize')
            sampleSize = 500;
        else
            sampleSize = Parameters.sampleSize;
        end
        if ~isfield(Parameters, 'sampleBounds')
            sampleBounds = [0, Info.maxIndex];
        else
            sampleBounds = Parameters.sampleBounds;
        end
        if ~isfield(Parameters, 'maxQuantile')
            maxQuantile = 0.95;
        else
            maxQuantile = Parameters.maxQuantile;
        end
    end
    
    disp(maxQuantile)
    % the first few frames always suck while the galvo winds up - exclude them
    offset = 10;
    
    if sampleBounds(1) > sampleBounds(2)
        sampleBounds = sampleBounds(end:-1:1);
    elseif sampleBounds(1) == sampleBounds(2)
        sampleBounds = [0, Info.maxIndex];
    end
    
    if sampleSize > sampleBounds(2) - sampleBounds(1) - offset
        sampleSize = sampleBounds(2) - sampleBounds(1) - offset;
    end
    
    motionCorrected = false;
    
    imageSize = Info.sz;
    
    % check if data has already been motion corrected
    if exist([Info.Directory.folder, Info.Directory.name, '.rigid'], 'file')
        load([Info.Directory.folder, Info.Directory.name, '.rigid'], '-mat', 'phaseDifferences', 'rowShifts', 'columnShifts', 'frameCrop');
        
        imageSize = [Info.sz(1) - frameCrop(3) - frameCrop(4), Info.sz(2) - frameCrop(1) - frameCrop(2)];
    
        motionCorrected = true;
    elseif exist([Info.Directory.folder, Info.Directory.name, '.align'], 'file')
        load([Info.Directory.folder, Info.Directory.name, '.align'], '-mat', 'phaseDifferences', 'rowShifts', 'columnShifts', 'frameCrop');
        
        imageSize = [Info.sz(1) - frameCrop(3) - frameCrop(4), Info.sz(2) - frameCrop(1) - frameCrop(2)];
    
        motionCorrected = true;
    end

    % create image stacks for cross correlation and PCA prediction
    imageStack = zeros(imageSize(1), imageSize(2), sampleSize);
    
%     if GUI
%         progressBar = waitbar(0, 'Generating image stack...', 'Name', [Info.Directory.name, ': sbxPreAnalysis'], 'CreateCancelBtn', 'setappdata(gcbf, ''Canceling'', 1)');
%         setappdata(progressBar, 'Canceling', 0);
%     end
    
    framePool = 0:Info.maxIndex;
    
    f = 1;
    
    while f <= sampleSize
        index = randi(Info.maxIndex + 1 - offset) - 1 + offset;

        if ismember(index, framePool)
            if motionCorrected
                frame = applyMotionCorrection(Info, index, frameCrop, phaseDifferences, rowShifts, columnShifts);
            else
                frame = sbxRead(Info, index);
            end
            
            if max(frame(:)) > threshold
                
                % remove frame from frame pool to keep each index unique
                framePool(framePool == index) = [];

                if resonantOffset ~= 0
                    frame = applyResonantOffset(frame, resonantOffset);
                end
        
                if gaussianFilter > 0.0
                    frame = imgaussfilt(frame, gaussianFilter);
                end
        
                imageStack(:, :, f) = frame;
                
                f = f + 1;

%                 if GUI
%                     if getappdata(progressBar, 'Canceling')
%                         delete(progressBar);
% 
%                         meanImage = 0;
%                         maxIntensityImage = 0;
%                         crossCorrelationImage = 0;
%                         localCrossCorrelationImage = 0;
%                         pcaImage = 0;
% 
%                         result = 'Canceled';
%                         return
%                     else
%                         waitbar(f/sampleSize, progressBar);
%                     end
%                 end
            else
                
                % if frame doesn't have anything above threshold, then remove it from the framePool
                framePool(framePool == index) = [];
            end
        end
            
        if isempty(framePool)
            warning(['Only ', int2str(f - 1), ' frames exceeded pixel intensity threshold.']);

            imageStack = imageStack(:, :, 1:f - 1);
            
            break
        end
    end
    
    % help out so memory doesn't get swamped
    if motionCorrected
        clearvars phaseDifferences rowShifts columnShifts adjustedImage Nr Nc
    end
        
    meanImage = squeeze(mean(imageStack, 3));
    
    maxIntensityImage = zeros(size(meanImage));
    primage50 = zeros(size(meanImage));
    primage75=primage50;
    primage90=primage50;
    
    for i = 1:size(meanImage, 1)
        for j = 1:size(meanImage, 2)
            maxIntensityImage(i, j) = quantile(imageStack(i, j, :), maxQuantile);
             primage50(i, j) = quantile(imageStack(i, j, :), 0.500);
             primage75(i, j) = quantile(imageStack(i, j, :), 0.750);
             primage90(i, j) = quantile(imageStack(i, j, :), 0.900);
        end
    end
    
    
    
        
    % clear up some memory
    clear imageStack
   
      
    % save individual images in handles structure
        save([Info.Directory.folder, Info.Directory.name, '.pre'], 'meanImage', 'maxIntensityImage', 'primage50', 'primage75', 'primage90');

%     delete(progressBar);
    
    result = 'Completed';

end



function adjustedImage = applyMotionCorrection(Info, f, frameCrop, phaseDifferences, rowShifts, columnShifts)

    frame = sbxRead(Info, f);

    if any(frameCrop > 0)
        frame = frame(frameCrop(3) + 1:Info.sz(1) - frameCrop(4), frameCrop(1) + 1:Info.sz(2) - frameCrop(2));
    end
    
    phaseDifference = phaseDifferences(f + 1);
    rowShift = rowShifts(f + 1);
    columnShift = columnShifts(f + 1);
    
    if phaseDifference ~= 0 || rowShift ~= 0 || columnShift ~= 0
        adjustedImage = fft2(frame);

        [numberOfRows, numberOfColumns] = size(adjustedImage);
        Nr = ifftshift(-fix(numberOfRows/2):ceil(numberOfRows/2) - 1);
        Nc = ifftshift(-fix(numberOfColumns/2):ceil(numberOfColumns/2) - 1);
        [Nc, Nr] = meshgrid(Nc, Nr);

        adjustedImage = adjustedImage.*exp(2i*pi*(-rowShift*Nr/numberOfRows - columnShift*Nc/numberOfColumns));
        adjustedImage = adjustedImage*exp(1i*phaseDifference);

        adjustedImage = abs(ifft2(adjustedImage));
        
        % adjust values just in case
        originalMinimum = double(min(frame(:)));
        originalMaximum = double(max(frame(:)));
        adjustedMinimum = min(adjustedImage(:));
        adjustedMaximum = max(adjustedImage(:));
        
        adjustedImage = uint16((adjustedImage - adjustedMinimum)/(adjustedMaximum - adjustedMinimum)*(originalMaximum - originalMinimum) + originalMinimum);
    else
        adjustedImage = frame;
    end
        
end

function adjustedImage = applyResonantOffset(frame, resonantOffset)

    adjustedImage = frame;

    if resonantOffset > 0
        for j = 2:2:size(frame, 1)
            adjustedImage(j, 1:end) = [zeros(1, resonantOffset), frame(j, 1:end - resonantOffset)];
        end
    elseif resonantOffset < 0
        for j = 2:2:size(frame, 1)
            adjustedImage(j, 1:end) = [frame(j, 1 + -resonantOffset:end), zeros(1, -resonantOffset)];
        end
    end
    
end