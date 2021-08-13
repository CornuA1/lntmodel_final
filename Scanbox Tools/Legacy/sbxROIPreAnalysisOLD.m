function sbxROIPreAnalysis(sbxName)

    %SBXROIPREANALYSIS Pre-compute mean images for a given dataset.
    %   SBXROIPREANALYSIS(sbxName) calculates mean images and cross correlations in an .sbx file and saves them in a separate file.
    %
    %   sbxName: string 
    %       Path of .sbx file to be analyzed (e.g., 'C:/User/xx0_000_001.sbx').
    %
    %   Original Authors: L. Fischer, Massachusetts Institute of Technology
    %                     J. Voigts, Massachusetts Institute of Technology
    
    % indicate cross-correlation method
    method = 1;

    % set sample size for mean image
    sampleSize = 1000;
    
    % set quantile to use for max intensity projection
    maxQuantile = 0.90;
    
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
        
    % read a sample of random images
    indices = randperm(Info.maxIndex + 1, sampleSize) - 1;

    % create sample image
    imageStack = zeros(Info.sz(1), Info.sz(2), sampleSize);
    
    for j = 1:length(indices)
        frame = sbxRead(Info, indices(j));
        imageStack(:, :, j) = frame;
    end
    
    meanReference = squeeze(mean(imageStack, 3));
    
    maxIntensityProjection = zeros(size(meanReference));
    
    for i = 1:size(meanReference, 1)
        for j = 1:size(meanReference, 2)
            maxIntensityProjection(i, j) = quantile(imageStack(i, j, :), maxQuantile);
        end
    end

    wGlobal = 1;
    w = 20;
    yMax = size(imageStack, 1);
    xMax = size(imageStack, 2);
    nFrames = size(imageStack, 3);
    ccImage = zeros(yMax, xMax);
    ccLocal = zeros(yMax, xMax, w*2 + 1, w*2 + 1);

    switch method
        
        % use labrigger method
        case 1
            disp('Processing global cross-correlation map...');
            
            % first calculate global cross-correlation map
            parfor y = 1 + wGlobal:yMax - wGlobal
                for x = 1 + wGlobal:xMax - wGlobal
                    
                    % extract center pixel's time course and subtract its mean
                    thing1 = reshape(imageStack(y, x, :) - mean(imageStack(y, x, :), 3), [1, 1, nFrames]); 
                    
                    % auto-correlation, for normalization later
                    ad_a = sum(thing1.*thing1, 3);

                    % extract the neighborhood
                    a = imageStack(y - wGlobal:y + wGlobal, x - wGlobal:x + wGlobal, :);   
                    
                    % get its mean and subtract it
                    b = mean(imageStack(y - wGlobal:y + wGlobal, x - wGlobal:x + wGlobal, :), 3);
                    thing2 = bsxfun(@minus, a, b);   
                    
                    % auto-correlation, for normalization later 
                    ad_b = sum(thing2.*thing2, 3);

                    % cross-correlation with normalization
                    crossCorrelation = sum(bsxfun(@times, thing1, thing2), 3)./sqrt(bsxfun(@times, ad_a, ad_b)); 
                    
                    % delete the middle point
                    crossCorrelation((numel(crossCorrelation) + 1)/2) = [];      
                    
                    % get the mean cross-correlation of the local neighborhood
                    ccImage(y, x) = mean(crossCorrelation(:));       
                end
            end      
            
            disp('Processing pixel-by-pixel cross-correlation map...');
            
            parfor y = 1 + w:yMax - w
                ccRow = zeros(xMax, w*2 + 1, w*2 + 1);
                
                for x = 1 + w:xMax - w
                    
                    % extract center pixel's time course and subtract its mean
                    thing1 = reshape(imageStack(y, x, :) - mean(imageStack(y, x, :), 3), [1, 1, nFrames]);
                    
                    % auto-correlation, for normalization later
                    ad_a = sum(thing1.*thing1, 3);

                    % extract the neighborhood
                    a = imageStack(y - w:y + w, x - w:x + w, :);
                    
                    % get its mean and subtract it
                    b = mean(imageStack(y - w:y + w, x - w:x + w, :), 3);
                    thing2 = bsxfun(@minus, a, b);
                    
                    % auto-correlation, for normalization later 
                    ad_b = sum(thing2.*thing2,3);

                    % cross-correlation with normalization
                    crossCorrelation = sum(bsxfun(@times, thing1, thing2), 3)./sqrt(bsxfun(@times, ad_a, ad_b));
                    
                    % get the mean cross-correlation of the local neighborhood
                    ccRow(x, :, :) = crossCorrelation;
                    
                    % delete the middle point
                    crossCorrelation((numel(crossCorrelation) + 1)/2) = [];
                end
                
                ccLocal(y, :, :, :) = ccRow;
            end
    end

    temp = mean(ccImage(:));
    ccImage(1, :) = temp;
    ccImage(end, :) = temp;
    ccImage(:, 1) = temp;
    ccImage(:, end) = temp;

    disp('Processing PCA ROI prediction...');
    
    % make PCA composite; this seems to display good ROI candidates
    vStack = zeros(nFrames, size(imageStack, 1)*size(imageStack, 2));
    
    for i = 1:nFrames
        x = imageStack(:, :, i);
        vStack(i, :) = x(:);
    end
    
    vStack = vStack - mean(imageStack(:));
    
    [coefficient, ~] = pca(vStack, 'Economy', 'on', 'NumComponents', 100);
    
    imageComponents = reshape(coefficient', 100, size(imageStack, 1), size(imageStack, 2));
    
    pcaImage = squeeze(mean(abs(imageComponents(1:100, :, :))));
    
    meanBrightness = zeros(Info.maxIndex + 1, 1);
    
    for i = 0:Info.maxIndex
        frame = sbxRead(Info, i);
        meanBrightness(i + 1) = mean(mean(frame));
    end
    
    % save individual images in handles structure
    try
        save([sbxPath, sbxName, '.rsc'], 'meanReference', 'maxIntensityProjection', 'ccImage', 'ccLocal', 'pcaImage', 'meanBrightness', '-append');
    catch
        save([sbxPath, sbxName, '.rsc'], 'meanReference', 'maxIntensityProjection', 'ccImage', 'ccLocal', 'pcaImage', 'meanBrightness', '-v7.3');
    end
    
    disp('Done.');

end