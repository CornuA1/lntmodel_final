function [meanImage, maxIntensityImage, crossCorrelationImage, localCrossCorrelationImage, pcaImage, result] = sbxPreAnalysisNew(Info, Parameters)

    %SBXPREANALYSIS Pre-compute mean images for a given dataset.
    %   [result, meanImage, maxIntensityImage, crossCorrelationImage, localCrossCorrelationImage, pcaImage] = SBXPREANALYSIS(Info, Parameters) calculates mean images and cross correlations in an .sbx file and saves them in a separate file.
    %
    %   Info: structure 
    %       Info structure generated by sbxInfo from corresponding .mat file.
    %
    %   Parameters: structure
    %       Optional input containing parameter specifications.
    %
    %   meanImage: array
    %       Array with dimensions (imageHeight, imageWidth).
    %
    %   maxIntensityImage: array
    %       Array with dimensions (imageHeight, imageWidth).
    %
    %   crossCorrelationImage: array
    %       Array with dimensions (imageHeight, imageWidth).
    %
    %   localCrossCorrelationImage: array
    %       Array with dimensions (imageHeight, imageWidth, 2*localWindow + 1, 2*localWindow + 1).
    %
    %   pcaImage: array
    %       Array with dimensions (imageHeight, imageWidth).
    %
    %   result: string
    %       Used by sbxAnalysis to confirm if function ran correctly or was cancelled.
    %
    %   Original Authors: L. Fischer, Massachusetts Institute of Technology
    %                     J. Voigts, Massachusetts Institute of Technology
    
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
    
    if GUI
        progressBar = waitbar(0, 'Generating image stack...', 'Name', [Info.Directory.name, ': sbxPreAnalysis'], 'CreateCancelBtn', 'setappdata(gcbf, ''Canceling'', 1)');
        setappdata(progressBar, 'Canceling', 0);
    end
    
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

                if GUI
                    if getappdata(progressBar, 'Canceling')
                        delete(progressBar);

                        meanImage = 0;
                        maxIntensityImage = 0;
                        crossCorrelationImage = 0;
                        localCrossCorrelationImage = 0;
                        pcaImage = 0;

                        result = 'Canceled';
                        return
                    else
                        waitbar(f/sampleSize, progressBar);
                    end
                end
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
    
    for i = 1:size(meanImage, 1)
        for j = 1:size(meanImage, 2)
            maxIntensityImage(i, j) = quantile(imageStack(i, j, :), maxQuantile);
        end
    end
    
    numberOfRows = size(imageStack, 1);
    numberOfColumns = size(imageStack, 2);
    
    if GUI
        if getappdata(progressBar, 'Canceling')
            delete(progressBar);
            
            meanImage = 0;
            maxIntensityImage = 0;
            crossCorrelationImage = 0;
            localCrossCorrelationImage = 0;
            pcaImage = 0;
            
            result = 'Canceled';
            return
        else
            waitbar(0, progressBar, 'Processing PCA ROI prediction...');
        end
    end
    
    % pca wants a matrix with rows corresponding to frames and columns corresponding to the columnwise elements of each frame, centered at zero relative to the mean of the entire stack
    [coefficient, ~] = pca(reshape(imageStack, [imageSize(1)*imageSize(2), sampleSize]).' - mean(imageStack(:)), 'Economy', 'on', 'NumComponents', 100);
    
    if GUI
        if getappdata(progressBar, 'Canceling')
            delete(progressBar);

            meanImage = 0;
            maxIntensityImage = 0;
            crossCorrelationImage = 0;
            localCrossCorrelationImage = 0;
            pcaImage = 0;

            result = 'Canceled';
            return
        else
            waitbar(0.5, progressBar);
        end
    end
                
    imageComponents = reshape(coefficient', 100, numberOfRows, numberOfColumns);
    
    pcaImage = squeeze(mean(abs(imageComponents(1:100, :, :))));
    
    if GUI
        if getappdata(progressBar, 'Canceling')
            delete(progressBar);
            
            meanImage = 0;
            maxIntensityImage = 0;
            crossCorrelationImage = 0;
            localCrossCorrelationImage = 0;
            pcaImage = 0;
            
            result = 'Canceled';
            return
        else
            waitbar(0, progressBar, 'Processing global and local cross-correlation maps...');
        end
    end

    switch method
        
        % use labrigger method
        case 1
        	globalWindow = 1;
            localWindow = 20;
            
            localCrossCorrelationImage = zeros(numberOfRows, numberOfColumns, 2*localWindow + 1, 2*localWindow + 1);
            crossCorrelationImage = zeros(numberOfRows, numberOfColumns);
            
            parfor y = 1:numberOfRows
                
                % it's necessary to split this computation up into rows to follow MATLAB's parfor rules
                localCrossCorrelationRow = NaN(numberOfColumns, 2*localWindow + 1, 2*localWindow + 1);
                globalCrossCorrelationRow = NaN(1, numberOfColumns);
                    
                yMin = max(y - localWindow, 1);
                yMax = min(y + localWindow, numberOfRows);
                
                for x = 1:numberOfColumns
                    
                    xMin = max(x - localWindow, 1);
                    xMax = min(x + localWindow, numberOfColumns);
                    
                    % making this a non-broadcast variable doesn't speed things up enough to be worth it
                    temp = imageStack(yMin:yMax, xMin:xMax, :);
                    
                    % extract pixel and neighborhood time courses and subtract their respective means
                    neighborhoodTimeCourse = bsxfun(@minus, temp, mean(temp, 3));
                    pixelTimeCourse = neighborhoodTimeCourse(y - yMin + 1, x - xMin + 1, :);
                    
                    % auto-correlation, for normalization later 
                    neighborhoodAutoCorrelation = sum(neighborhoodTimeCourse.*neighborhoodTimeCourse, 3);
                    pixelAutoCorrelation = neighborhoodAutoCorrelation(y - yMin + 1, x - xMin + 1);

                    % cross-correlation with normalization
                    localCrossCorrelation = sum(bsxfun(@times, pixelTimeCourse, neighborhoodTimeCourse), 3)./sqrt(pixelAutoCorrelation*neighborhoodAutoCorrelation);
                                        
                    localCrossCorrelationRow(x, yMin - y + localWindow + 1:yMax - y + localWindow + 1, xMin - x + localWindow + 1:xMax - x + localWindow + 1) = localCrossCorrelation;
                    
                    % just use neighboring pixels for the global cross-correlation
                    globalCrossCorrelation = localCrossCorrelationRow(x, localWindow + 1 - globalWindow:localWindow + 1 + globalWindow, localWindow + 1 - globalWindow:localWindow + 1 + globalWindow);
                    
                    % delete the middle point
                    globalCrossCorrelation(globalWindow + 1, globalWindow + 1) = NaN;    
                    
                    % then take the mean, excluding NaNs from center and edge cases
                    globalCrossCorrelationRow(x) = nanmean(globalCrossCorrelation(:));
                end
                
                localCrossCorrelationImage(y, :, :, :) = localCrossCorrelationRow;
                crossCorrelationImage(y, :) = globalCrossCorrelationRow;
            end
    end
    
    % clear up some memory
    clear imageStack
    
    if GUI
        if getappdata(progressBar, 'Canceling')
            delete(progressBar);

            meanImage = 0;
            maxIntensityImage = 0;
            crossCorrelationImage = 0;
            localCrossCorrelationImage = 0;
            pcaImage = 0;

            result = 'Canceled';
            return
        else
            waitbar(0, progressBar, 'Calculating mean brightness...');
        end
    end
    
    if GUI
        if getappdata(progressBar, 'Canceling')
            delete(progressBar);
            
            meanImage = 0;
            maxIntensityImage = 0;
            crossCorrelationImage = 0;
            localCrossCorrelationImage = 0;
            pcaImage = 0;
            
            result = 'Canceled';
            return
        else
            waitbar(1, progressBar, 'Saving computations...');
        end
    end
    
    % save individual images in handles structure
    try
        save([Info.Directory.folder, Info.Directory.name, '.pre'], 'meanImage', 'maxIntensityImage', 'crossCorrelationImage', 'localCrossCorrelationImage', 'pcaImage', '-append');
    catch
        save([Info.Directory.folder, Info.Directory.name, '.pre'], 'meanImage', 'maxIntensityImage', 'crossCorrelationImage', 'localCrossCorrelationImage', 'pcaImage', '-v7.3');
    end
        
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