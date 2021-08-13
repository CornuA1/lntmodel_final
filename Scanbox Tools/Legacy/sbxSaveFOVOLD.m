function sbxSaveFOV(sbxName)

    %SBXSAVEFOV Generate a mean image of imaging data.
    %   SBXSAVEFOV(sbxName) creates a mean image of the selected imaging session using a defined number of randomly selected frames.
    %
    %   sbxName: string 
    %       Path of .sbx file to be analyzed (e.g., 'C:/User/xx0_000_001.sbx').

    % set sample size for mean image
    sampleSize = 500;
    
    roiMask = false;
    
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
    %indices = linspace(0, 3, 4);
    
    % create sample image
    imageStack = zeros(Info.sz(1), Info.sz(2), sampleSize);
    
    for j = 1:length(indices)
        frame = sbxRead(Info, indices(j));
        imageStack(:, :, j) = frame;
    end
    
    meanReference = squeeze(mean(imageStack, length(size(imageStack))));
    
    FOV = figure('Name', 'ROI FOV', 'Visible', 'off', 'Units', 'pixels', 'Position', [0, 0, Info.sz(2), Info.sz(1)], 'NumberTitle', 'off', 'Color', 'w');
    imageAxes = axes('Units', 'normalized', 'Position', [0.0, 0.0, 1.0, 1.0]);
    
    % you have to do this after axes initialization and adding any image
    axis(imageAxes, 'tight');
    axis(imageAxes, 'off');
    
    colormap(imageAxes, gray);
    
    %meanReference = double(meanReference)/double(intmax('uint16'));
    meanReference = double(meanReference)/double(max(meanReference(:)));
    
    imagesc(meanReference, [0, 1]);

    if roiMask
        try
            [roiMaskName, roiMaskPath] = uigetfile('.segment', 'Please select file containing ROI mask.');
        catch
            waitfor(msgbox('Error: Please select valid .segment file.'));
            error('Please select valid .segment file.');
        end   

        hold(imageAxes, 'on');
        
        roiImage = image(imageAxes, bsxfun(@times, ones(Info.sz(1), Info.sz(2)), reshape([0, 0, 1], [1, 1, 3])));

        hold(imageAxes, 'off');
        
        axis(imageAxes, 'tight');
        axis(imageAxes, 'off');

        roiMaskName = strtok(roiMaskName, '.');

        load([roiMaskPath, roiMaskName, '.segment'], '-mat');

        roiImage.AlphaData = 0.5*(roiMask > 0);
    end
    
    print(FOV, [sbxPath, sbxName, '_FOV'], '-dtiff');
    
    close(FOV);
    
end