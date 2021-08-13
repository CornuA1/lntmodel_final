function sbxROIAnalysis()

    %SBXROIANALYSIS Graphic interface for analyzing calcium imaging data.
    %   SBXROIANALYSIS() creates a graphic interface for analyzing ROI signals from calcium imaging data in .sbx format.
    %
    %   Original Author: L. Fischer, Massachusetts Institute of Technology

    GUI = figure('Name', 'ROI Analysis', 'Units', 'normalized', 'Position', [0.125, 0.125, 0.75, 0.75], 'MenuBar', 'none', 'NumberTitle', 'off', 'Color', 'w');

    zoom(GUI, 'off');
    pan(GUI, 'off');
    
    % call handles of figure so we can modify and add to them
    GUIHandles = guihandles(GUI);
    
    GUIHandles.parentFigure = GUI;
    
    % specify number of frames on each side for rolling average
    GUIHandles.rollingWindowSize = 2;
            
    % set minimum size of isolated region in ROI
    GUIHandles.defaultMinIslandSize = 30;
    GUIHandles.minIslandSize = GUIHandles.defaultMinIslandSize;
    GUIHandles.maxMinIslandSize = 100;
        
    % configure neuropil size
    GUIHandles.neuropilSize = 7;
    GUIHandles.maxNeuropilSize = 15;
    
    % set display adjustments
    GUIHandles.brightness = 0.0;
    GUIHandles.contrast = 1.0;

    % configure neuropil correction weight
    GUIHandles.neuropilCorrection = 1.0;
    
    % set default setting for number of pixels to include in flood filling
    GUIHandles.defaultFloodPixels = 250;
    GUIHandles.floodPixels = GUIHandles.defaultFloodPixels;
    
    % set default pixel intensity below which to remove pixels;
    GUIHandles.pruneThreshold = 0;
    
    GUIHandles.imageAxes = axes('Units', 'normalized', 'Position', [0.1, 0.0, 0.8, 1.0]);
    
    % you have to do this after axes initialization and adding any image
    axis(GUIHandles.imageAxes, 'tight');
    axis(GUIHandles.imageAxes, 'off');
    
    colormap(GUIHandles.imageAxes, gray);
       
    GUIHandles.loadSBXButton = uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.013, 0.92, 0.074, 0.05], 'String', 'Load Imaging Data', 'Callback', @loadSBXCallback);
    
    GUIHandles.statusText = uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.01, 0.02, 0.08, 0.0175], 'String', 'Ready', 'BackgroundColor', 'w');

    % save handles to figure
    guidata(GUI, GUIHandles);
    
end

function loadSBXCallback(GUI, ~)     
    
    try
        [sbxName, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.');
    catch
        waitfor(msgbox('Error: Please select valid .sbx file.'));
        error('Please select valid .sbx file.');
    end
    
    % pull off the file extension
    sbxName = strtok(sbxName, '.');
    
    Info = sbxInfo([sbxPath, sbxName]);
    
    if exist([sbxPath, sbxName, '.rsc'], 'file')
        GUIHandles = guidata(GUI);

        % make load data button invisible to replace with load new data button
        GUIHandles.loadSBXButton.Visible = 'off';
        
        GUIHandles.statusText.String = 'Loading...';
        drawnow;
    else
        waitfor(msgbox('Error: .rsc file containing pre-computed cross-correlations required for ROI analysis.'));
        error('.rsc file containing pre-computed cross-correlations required for ROI analysis.');
    end
   
    temp = who('-file', [sbxPath, sbxName, '.rsc']);
    
    if ismember('meanReference', temp)
        load([sbxPath, sbxName, '.rsc'], '-mat', 'meanReference', 'maxIntensityProjection', 'ccImage', 'ccLocal', 'pcaImage', 'meanBrightness')
        
        GUIHandles.meanReference = meanReference/double(intmax('uint16'));
        GUIHandles.maxIntensityProjection = maxIntensityProjection/double(intmax('uint16'));        
        GUIHandles.ccImage = ccImage;
        GUIHandles.ccLocal = ccLocal;
        GUIHandles.pcaImage = pcaImage/max(pcaImage(:));
        GUIHandles.meanBrightness = meanBrightness;
    else
        load([sbxPath, sbxName, '.rsc'], '-mat', 'meanref', 'ccimage', 'cc_local', 'pcaimage', 'mean_brightness')
        
        GUIHandles.meanReference = meanref;
        GUIHandles.ccImage = ccimage;
        GUIHandles.pcaImage = pcaimage;
        GUIHandles.ccLocal = cc_local;
        GUIHandles.meanBrightness = mean_brightness;
    end
    
    % save details about the .sbx file
    GUIHandles.sbxPath = sbxPath;
    GUIHandles.sbxName = sbxName;
    GUIHandles.Info = Info;
    
    GUIHandles.zoomCenter = [];
    
    % initialize currently selected ROI
    GUIHandles.selectedROI = 0;
                    
    % set the starting settings
    GUIHandles.imageType = 'Mean';
    GUIHandles.manipulateType = 'Draw';
    GUIHandles.drawType = 'Pixel';
        
    % initialize various masks
    GUIHandles.roiMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
    GUIHandles.selectedROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
    GUIHandles.previousROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
    
    GUIHandles.displayedImage = imagesc(GUIHandles.meanReference, [0, 1]);
    
    hold(GUIHandles.imageAxes, 'on');
    
    % initialize the different ROI images that will be overlaid onto the calcium images
    GUIHandles.roiImage = image(GUIHandles.imageAxes, bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([0, 0, 1], [1, 1, 3])));
    GUIHandles.candidateROIImage = image(GUIHandles.imageAxes, bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([1, 0, 0], [1, 1, 3])));
    GUIHandles.selectedROIImage = image(GUIHandles.imageAxes, bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([0, 1, 0], [1, 1, 3])));
    GUIHandles.neuropilImage = image(GUIHandles.imageAxes, bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([0.25, 0.75, 0], [1, 1, 3])));
    GUIHandles.islandImage = image(GUIHandles.imageAxes, bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([1, 0, 0], [1, 1, 3])));
    
    hold(GUIHandles.imageAxes, 'off');
    
    GUIHandles.candidateROIImage.Visible = 'off';
    GUIHandles.selectedROIImage.Visible = 'off';
    GUIHandles.neuropilImage.Visible = 'off';
    GUIHandles.islandImage.Visible = 'off';
    
    axis(GUIHandles.imageAxes, 'tight');
    axis(GUIHandles.imageAxes, 'off');
    
    GUIHandles.XLim = GUIHandles.imageAxes.XLim;
    GUIHandles.YLim = GUIHandles.imageAxes.YLim;
    GUIHandles.maxWindowSize = (GUIHandles.XLim(2) - GUIHandles.XLim(1))/2;
    GUIHandles.windowSize = GUIHandles.maxWindowSize;
    GUIHandles.windowRatio = (GUIHandles.YLim(2) - GUIHandles.YLim(1))/(GUIHandles.XLim(2) - GUIHandles.XLim(1));
    
    % make all of the UI controls. basic controls on the left side
    GUIHandles.loadNewSBXButton = uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.013, 0.93, 0.074, 0.05], 'String', 'Load New Data', 'Callback', @loadNewSBXCallback);
    
    GUIHandles.imageTypeButtonGroup = uibuttongroup('Title', 'Display', 'TitlePosition', 'centertop', 'Units', 'normalized', 'Position', [0.0075, 0.815, 0.085, 0.15], 'BackgroundColor', 'w', 'SelectionChangedFcn', @imageTypeButtonGroupSelectionFunction);
    GUIHandles.meanButton = uicontrol('Parent', GUIHandles.imageTypeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.05, 5.0/6.0, 0.9, 1.0/6.0], 'String', 'Mean', 'BackgroundColor', 'w');
    GUIHandles.maxButton = uicontrol('Parent', GUIHandles.imageTypeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.05, 4.0/6.0, 0.9, 1.0/6.0], 'String', 'Max Intensity', 'BackgroundColor', 'w');
    GUIHandles.ccButton = uicontrol('Parent', GUIHandles.imageTypeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.05, 3.0/6.0, 0.9, 1.0/6.0], 'String', 'Cross-Correlation', 'BackgroundColor', 'w');
    GUIHandles.pcaButton = uicontrol('Parent', GUIHandles.imageTypeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.05, 2.0/6.0, 0.9, 1.0/6.0], 'String', 'PCA', 'BackgroundColor', 'w');
    GUIHandles.rollingButton = uicontrol('Parent', GUIHandles.imageTypeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.05, 1.0/6.0, 0.9, 1.0/6.0], 'String', 'Rolling Average', 'BackgroundColor', 'w');
    GUIHandles.frameButton = uicontrol('Parent', GUIHandles.imageTypeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.05, 0.0, 0.9, 1.0/6.0], 'String', 'Rolling Average', 'BackgroundColor', 'w');
    
    GUIHandles.frameSlider = uicontrol('Style', 'slider', 'Visible', 'off', 'Min', 0, 'Max', Info.maxIndex, 'Value', 0, 'Units', 'normalized', 'Position', [0.0075, 0.8, 0.085, 0.015], 'SliderStep', [1/Info.maxIndex, 50/Info.maxIndex], 'Callback', @frameSliderCallback);
 
    GUIHandles.ccLocalButton = uicontrol('Style', 'checkbox', 'Units', 'normalized', 'Position', [0.0025, 0.775, 0.095, 0.025], 'String', 'Local Cross-Correlation', 'BackgroundColor', 'w', 'Callback', @ccLocalButtonCallback);

    GUIHandles.loadROIMaskButton = uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.016, 0.710, 0.068, 0.05], 'String', 'Load ROI Mask', 'Callback', @loadROIMaskCallback);
    GUIHandles.newROIMaskButton = uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.016, 0.66, 0.068, 0.05], 'String', 'New ROI Mask', 'Callback', @newROIMaskCallback);
    
    GUIHandles.extractFluorescencesButton = uicontrol('Style', 'pushbutton', 'Interruptible', 'on', 'Units', 'normalized', 'Position', [0.009, 0.605, 0.082, 0.04], 'String', 'Extract Fluorescences', 'Callback', @extractFluorescencesCallback);
    GUIHandles.cancelExtractionButton = uicontrol('Style', 'pushbutton', 'Visible', 'off', 'Units', 'normalized', 'Position', [0.025, 0.605, 0.05, 0.04], 'String', 'Cancel', 'Callback', @cancelExtractionCallback);
    GUIHandles.undoButton = uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.025, 0.565, 0.05, 0.04], 'String', 'Undo', 'Callback', @undoCallback);
    GUIHandles.redoButton = uicontrol('Style', 'pushbutton', 'Visible', 'off', 'Units', 'normalized', 'Position', [0.025, 0.565, 0.05, 0.04], 'String', 'Redo', 'Callback', @undoCallback);
    GUIHandles.deleteROIsButton = uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.025, 0.525, 0.05, 0.04], 'String', 'Delete ROI', 'Callback', @deleteROIsCallback);
    GUIHandles.saveROIsButton = uicontrol('Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.025, 0.485, 0.05, 0.04], 'String', 'Save ROIs', 'Callback', @saveROIsCallback);
        
    GUIHandles.progressBarAxes = axes('Units', 'normalized', 'Position', [0.01, 0.013, 0.08, 0.0075]);
    GUIHandles.progressBar = histogram(GUIHandles.progressBarAxes, [], linspace(0, 100, 100), 'Visible', 'off', 'FaceColor', 'k', 'EdgeColor', 'none');
    GUIHandles.progressBarAxes.Visible = 'off';
    GUIHandles.progressBarAxes.XLim = [0, 100];
    GUIHandles.progressBarAxes.YLim = [0, 1];
    set(GUIHandles.progressBarAxes, 'xtick', []);
    set(GUIHandles.progressBarAxes, 'ytick', []);
    set(GUIHandles.progressBarAxes, 'xticklabel', []);
    set(GUIHandles.progressBarAxes, 'yticklabel', []);
    
    % analysis tools on the right side
    GUIHandles.manipulateROIsButtonGroup = uibuttongroup('Title', 'ROI Manipulation', 'TitlePosition', 'centertop', 'Units', 'normalized', 'Position', [0.9125, 0.395, 0.075, 0.075], 'BackgroundColor', 'w', 'SelectionChangedFcn', @manipulateROIsButtonGroupSelectionFunction);
    GUIHandles.drawButton = uicontrol('Parent', GUIHandles.manipulateROIsButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.25, 2/3, 0.5, 1/3], 'String', 'Draw', 'BackgroundColor', 'w');
    GUIHandles.selectButton = uicontrol('Parent', GUIHandles.manipulateROIsButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.25, 1/3, 0.5, 1/3], 'String', 'Select', 'BackgroundColor', 'w');
    GUIHandles.pruneButton = uicontrol('Parent', GUIHandles.manipulateROIsButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.25, 0.0, 0.5, 1/3], 'String', 'Prune', 'BackgroundColor', 'w');
        
    GUIHandles.drawShapeButtonGroup = uibuttongroup('Title', 'ROI Draw Method', 'TitlePosition', 'centertop', 'Units', 'normalized', 'Position', [0.9125, 0.255, 0.075, 0.125], 'BackgroundColor', 'w', 'SelectionChangedFcn', @drawTypeButtonGroupSelectionFunction);
    GUIHandles.floodfillButton = uicontrol('Parent', GUIHandles.drawShapeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.15, 0.8, 0.7, 0.2], 'String', 'Pixel', 'BackgroundColor', 'w');
    GUIHandles.freeHandButton = uicontrol('Parent', GUIHandles.drawShapeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.15, 0.6, 0.7, 0.2], 'String', 'Free Hand', 'BackgroundColor', 'w');
    GUIHandles.polygonButton = uicontrol('Parent', GUIHandles.drawShapeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.15, 0.4, 0.7, 0.2], 'String', 'Polygon', 'BackgroundColor', 'w');
    GUIHandles.ellipseButton = uicontrol('Parent', GUIHandles.drawShapeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.15, 0.2, 0.7, 0.2], 'String', 'Ellipse', 'BackgroundColor', 'w');
    GUIHandles.rectangleButton = uicontrol('Parent', GUIHandles.drawShapeButtonGroup, 'Style', 'radiobutton', 'Units', 'normalized', 'Position', [0.15, 0.0, 0.7, 0.2], 'String', 'Rectangle', 'BackgroundColor', 'w');

    GUIHandles.leftButton = uicontrol('Style', 'pushbutton', 'Visible', 'Off', 'Units', 'normalized', 'Position', [0.9125, 0.33, 0.025, 0.025], 'String', 'Left', 'Callback', @moveCallback);
    GUIHandles.rightButton = uicontrol('Style', 'pushbutton', 'Visible', 'Off', 'Units', 'normalized', 'Position', [0.9625, 0.33, 0.025, 0.025], 'String', 'Right', 'Callback', @moveCallback);
    GUIHandles.upButton = uicontrol('Style', 'pushbutton', 'Visible', 'Off', 'Units', 'normalized', 'Position', [0.9375, 0.355, 0.025, 0.025], 'String', 'Up', 'Callback', @moveCallback);
    GUIHandles.downButton = uicontrol('Style', 'pushbutton', 'Visible', 'Off', 'Units', 'normalized', 'Position', [0.9375, 0.305, 0.025, 0.025], 'String', 'Down', 'Callback', @moveCallback);
    
    GUIHandles.findROIButton = uicontrol('Style', 'pushbutton', 'Visible', 'off', 'Units', 'normalized', 'Position', [0.9275, 0.25, 0.045, 0.04], 'String', 'Find ROI', 'Callback', @findROICallback);
    
    GUIHandles.fillROIButton = uicontrol('Style', 'pushbutton', 'Visible', 'off', 'Units', 'normalized', 'Position', [0.9275, 0.195, 0.045, 0.04], 'String', 'Fill ROI', 'Callback', @fillROICallback);
    GUIHandles.clusterROIsButton = uicontrol('Style', 'pushbutton', 'Visible', 'off', 'Units', 'normalized', 'Position', [0.925, 0.195, 0.05, 0.04], 'String', 'Cluster ROIs', 'Callback', @clusterROIsCallback);
    
    GUIHandles.histogramAxes = axes('Units', 'normalized', 'Position', [0.9175, 0.254, 0.065, 0.125]);
    GUIHandles.histogramPlot = histogram(GUIHandles.histogramAxes, [], 'Visible', 'off', 'FaceColor', 'k');
    GUIHandles.histogramAxes.XLim = [0, 1];
    GUIHandles.histogramAxes.YLim = [0, 1];    
    GUIHandles.histogramAxes.XLabel.String = 'Pixel Intensity';
    GUIHandles.histogramAxes.FontName = 'MS Sans Serif';
    GUIHandles.histogramAxes.FontSize = 8;
    GUIHandles.histogramAxes.Visible = 'off';
    
    GUIHandles.predictFluorescenceButton = uicontrol('Style', 'pushbutton', 'Visible', 'off', 'Interruptible', 'on', 'Units', 'normalized', 'Position', [0.909, 0.1655, 0.082, 0.04], 'String', 'Predict Fluorescence', 'Callback', @predictFluorescenceCallback);
    GUIHandles.cancelPredictionButton = uicontrol('Style', 'pushbutton', 'Visible', 'off', 'Units', 'normalized', 'Position', [0.925, 0.1655, 0.05, 0.04], 'String', 'Cancel', 'Callback', @cancelPredictionCallback);
    
    GUIHandles.islandSizeText = uicontrol('Style', 'text', 'Visible', 'off', 'Units', 'normalized', 'Position', [0.9075, 0.1445, 0.085, 0.0175], 'String', 'Minimum Region Size', 'BackgroundColor', 'w');
    GUIHandles.islandSizeSlider = uicontrol('Style', 'slider', 'Visible', 'off', 'Min', 0, 'Max', GUIHandles.maxMinIslandSize, 'Value', GUIHandles.minIslandSize, 'Units', 'normalized', 'Position', [0.9075, 0.127, 0.085, 0.015], 'SliderStep', [1/GUIHandles.maxMinIslandSize, 50/GUIHandles.maxMinIslandSize], 'Callback', @islandSizeSliderCallback);

    GUIHandles.viewNeuropilButton = uicontrol('Style', 'checkbox', 'Visible', 'Off', 'Units', 'normalized', 'Position', [0.9125, 0.0985, 0.075, 0.025], 'String', 'View Neuropil', 'BackgroundColor', 'w', 'Callback', @viewNeuropilButtonCallback);
        
    GUIHandles.neuropilSizeText = uicontrol('Style', 'text', 'Visible', 'off', 'Units', 'normalized', 'Position', [0.9075, 0.0835, 0.085, 0.0175], 'String', 'Neuropil Size', 'BackgroundColor', 'w');
    GUIHandles.neuropilSizeSlider = uicontrol('Style', 'slider', 'Visible', 'off', 'Min', 1, 'Max', GUIHandles.maxNeuropilSize, 'Value', GUIHandles.neuropilSize, 'Units', 'normalized', 'Position', [0.9075, 0.066, 0.085, 0.015], 'SliderStep', [1/GUIHandles.maxNeuropilSize, 5/GUIHandles.maxNeuropilSize], 'Callback', @neuropilSizeSliderCallback);
    
    GUIHandles.brightnessText = uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.9075, 0.81, 0.085, 0.0175], 'String', 'Brightness', 'BackgroundColor', 'w');
    GUIHandles.brightnessSlider = uicontrol('Style', 'slider', 'Min', -0.9, 'Max', 0.9, 'Value', GUIHandles.brightness, 'Units', 'normalized', 'Position', [0.9075, 0.8, 0.085, 0.015], 'SliderStep', [0.01, 0.2], 'Callback', @brightnessSliderCallback);
    
    GUIHandles.contrastText = uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.9075, 0.71, 0.085, 0.0175], 'String', 'Contrast', 'BackgroundColor', 'w');
    GUIHandles.contrastSlider = uicontrol('Style', 'slider', 'Min', 0.5, 'Max', 10.0, 'Value', GUIHandles.contrast, 'Units', 'normalized', 'Position', [0.9075, 0.7, 0.085, 0.015], 'SliderStep', [0.05, 0.25], 'Callback', @contrastSliderCallback);
    
    % attach customized functions to figure
    set(GUIHandles.parentFigure, 'ButtonDownFcn', @roiImageClickFunction);
    set(GUIHandles.parentFigure, 'WindowButtonMotionFcn', @mouseOverFunction);
    set(GUIHandles.parentFigure, 'WindowScrollWheelFcn', @scrollFunction);
    set(GUIHandles.parentFigure, 'KeyPressFcn', @keyPressFunction);
    set(GUIHandles.parentFigure, 'KeyReleaseFcn', @keyReleaseFunction);
    
    % attach click callbacks to images and masks
    set(GUIHandles.roiImage, 'ButtonDownFcn', @roiImageClickFunction);
    set(GUIHandles.candidateROIImage, 'ButtonDownFcn', @candidateROIImageClickFunction);
    set(GUIHandles.selectedROIImage, 'ButtonDownFcn', @selectedROIImageClickFunction);
    set(GUIHandles.neuropilImage, 'ButtonDownFcn', @selectedROIImageClickFunction);
    set(GUIHandles.islandImage, 'ButtonDownFcn', @selectedROIImageClickFunction);
    
    GUIHandles.statusText.String = 'Ready';
    
    guidata(GUI, GUIHandles);
    
    updateImageDisplay(GUI);
    updateROIImage(GUI);

end

function loadNewSBXCallback(GUI, ~) 
    
    try
        [sbxName, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.');
    catch
        waitfor(msgbox('Error: Please select valid .sbx file.'));
        error('Please select valid .sbx file.');
    end
    
    sbxName = strtok(sbxName, '.');
    
    Info = sbxInfo([sbxPath, sbxName]);
    
    if exist([sbxPath, sbxName, '.rsc'], 'file')        
        GUIHandles = guidata(GUI);

        GUIHandles.displayedImage.CData = 1;
        
        GUIHandles.statusText.String = 'Loading...';
        drawnow;
   
        temp = who('-file', [sbxPath, sbxName, '.rsc']);

        if ismember('meanReference', temp)
            load([sbxPath, sbxName, '.rsc'], '-mat', 'meanReference', 'maxIntensityProjection', 'ccImage', 'ccLocal', 'pcaImage', 'meanBrightness')

            GUIHandles.meanReference = meanReference/double(intmax('uint16'));
            GUIHandles.maxIntensityProjection = maxIntensityProjection/double(intmax('uint16'));        
            GUIHandles.ccImage = ccImage;
            GUIHandles.ccLocal = ccLocal;
            GUIHandles.pcaImage = pcaImage/max(pcaImage(:));
            GUIHandles.meanBrightness = meanBrightness;
        else
            load([sbxPath, sbxName, '.rsc'], '-mat', 'meanref', 'ccimage', 'cc_local', 'pcaimage', 'mean_brightness')

            GUIHandles.meanReference = meanref;
            GUIHandles.ccImage = ccimage;
            GUIHandles.pcaImage = pcaimage;
            GUIHandles.ccLocal = cc_local;
            GUIHandles.meanBrightness = mean_brightness;
        end
        
        GUIHandles.statusText.String = 'Ready';
    else
        waitfor(msgbox('Error: .rsc file containing pre-computed cross-correlations required for ROI analysis.'));
        error('.rsc file containing pre-computed cross-correlations required for ROI analysis.');
    end
        
    GUIHandles.sbxPath = sbxPath;
    GUIHandles.sbxName = sbxName;
    GUIHandles.Info = Info;
    
    GUIHandles.zoomCenter = [];
    
    GUIHandles.selectedROI = 0;
    
    GUIHandles.imageAxes.XLim = GUIHandles.XLim;
    GUIHandles.imageAxes.YLim = GUIHandles.YLim;
    
    % these values have changed
    GUIHandles.frameSlider.Max = Info.maxIndex;
    GUIHandles.frameSlider.Value = 0;
    GUIHandles.SliderStep = [1/Info.maxIndex, 50/Info.maxIndex];
    
    guidata(GUI, GUIHandles);
    
    updateImageDisplay(GUI);
    
end

function imageTypeButtonGroupSelectionFunction(GUI, eventdata)

    GUIHandles = guidata(GUI);
    
    % respond to changes in selected image type
    GUIHandles.imageType = get(eventdata.NewValue, 'String');
    
    switch GUIHandles.imageType
        case 'Mean'
            GUIHandles.frameSlider.Visible = 'off';
        case 'Max Intensity'
            GUIHandles.frameSlider.Visible = 'off';
        case 'Cross-Correlation'
            GUIHandles.frameSlider.Visible = 'off';
        case 'PCA'
            GUIHandles.frameSlider.Visible = 'off';
        case 'Rolling Average'
            GUIHandles.frameSlider.Visible = 'on';
        case 'Frame-by-Frame'
            GUIHandles.frameSlider.Visible = 'on';
    end
    
    guidata(GUI, GUIHandles);
    
    updateImageDisplay(GUI);

end

function frameSliderCallback(GUI, ~)

    updateImageDisplay(GUI);

end

function ccLocalButtonCallback(GUI, ~)

    GUIHandles = guidata(GUI);
    
    % make sure to get rid of local cross-correlation square when turned off
    if GUIHandles.ccLocalButton.Value == 0        
        updateImageDisplay(GUI);
    end

end

function loadROIMaskCallback(GUI, ~)  
    
    try
        [roiMaskName, roiMaskPath] = uigetfile('.segment', 'Please select file containing ROI mask.');
    catch
        waitfor(msgbox('Error: Please select valid .segment file.'));
        error('Please select valid .segment file.');
    end   
    
    roiMaskName = strtok(roiMaskName, '.');
    
    load([roiMaskPath, roiMaskName, '.segment'], '-mat');

    GUIHandles = guidata(GUI);
    
    try
        GUIHandles.roiMask = roiMask;
    catch
        GUIHandles.roiMask = mask;
    end
    
    GUIHandles.selectedROIMask = zeros(size(GUIHandles.roiMask));
    GUIHandles.previousROIMask = zeros(size(GUIHandles.roiMask));
    
    GUIHandles.roiImage.CData = bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([0, 0, 1], [1, 1, 3]));
    GUIHandles.candidateROIImage.CData = bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([1, 0, 0], [1, 1, 3]));
    GUIHandles.selectedROIImage.CData = bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([0, 1, 0], [1, 1, 3]));
    GUIHandles.neuropilImage.CData = bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([0.25, 0.75, 0], [1, 1, 3]));
    GUIHandles.islandImage.CData = bsxfun(@times, ones(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)), reshape([1, 0, 0], [1, 1, 3]));
    
    GUIHandles.roiMaskPath = roiMaskPath;
    GUIHandles.roiMaskName = roiMaskName;    
    
    guidata(GUI, GUIHandles);
    
    updateROIImage(GUI);
    updateSelectedROIImage(GUI);
    
end

function newROIMaskCallback(GUI, ~)

    GUIHandles = guidata(GUI);
    
    GUIHandles.roiMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
    GUIHandles.selectedROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
    GUIHandles.previousROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
    
    GUIHandles.roiMaskPath = GUIHandles.sbxPath;
    GUIHandles.roiMaskName = GUIHandles.sbxName;
    
    guidata(GUI, GUIHandles);
    
    updateROIImage(GUI);
    updateSelectedROIImage(GUI);
    
end

function extractFluorescencesCallback(GUI, ~)
    
    GUIHandles = guidata(GUI);
    
    % make sure no ROIs are selected
    if GUIHandles.selectedROI(1) > 0
        selectedROIImageClickFunction(GUI);
    end
    
    % have to update GUIHandles again
    GUIHandles = guidata(GUI);
    
    if nnz(GUIHandles.roiMask) > 0
        numberOfROIs = max(GUIHandles.roiMask(:));
        numberOfFrames = GUIHandles.Info.maxIndex + 1;
        
        if GUIHandles.neuropilCorrection > 0
            neuropilMasks = cell(1, numberOfROIs);
            
            allROIsMask = GUIHandles.roiMask > 0;
            
            for r = 1:numberOfROIs
                currentROI = GUIHandles.roiMask == r;

                % we dilate the ROI with a disk structure to create the neuropil. note the subtraction of all pixels belonging to ROIs!
                if nnz(currentROI) > 0
                    neuropilMasks{r} = (imdilate(currentROI, strel('disk', GUIHandles.neuropilSize)) - allROIsMask) > 0;
                else
                    neuropilMasks{r} = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
                end
            end

            neuropilValues = zeros(numberOfFrames, numberOfROIs);
            correctedROIValues = zeros(numberOfFrames, numberOfROIs);
        end

        roiValues = zeros(numberOfFrames, numberOfROIs);

        roiMask = GUIHandles.roiMask;
        neuropilCorrection = GUIHandles.neuropilCorrection;
        frameSize = GUIHandles.Info.sz;
        
        frames = zeros(numberOfFrames, frameSize(1), frameSize(2), 'uint16');
        
        GUIHandles.progressBarAxes.Visible = 'on';
        GUIHandles.progressBar.Visible = 'on';

        GUIHandles.extractFluorescencesButton.Visible = 'off';
        GUIHandles.cancelExtractionButton.Visible = 'on';
        
        GUIHandles.statusText.String = 'Reading Frames...';
        drawnow;
        
        % do this first so you don't have to do it for every ROI
        for f = 1:numberOfFrames
            frames(f, :, :) = sbxRead(GUIHandles.Info, f - 1);
            
            GUIHandles.progressBar.Data = linspace(0, round(f/numberOfFrames*100), round(f/numberOfFrames*100));
            drawnow;
        end
        
        GUIHandles.progressBar.Data = [];
        
        GUIHandles.statusText.String = 'Extracting Fluorescences...';
        drawnow;
        
        for r = 1:numberOfROIs
            GUIHandles.progressBar.Data = linspace(0, round(r/numberOfROIs*100), round(r/numberOfROIs*100));
            drawnow;
            
            currentROI = roiMask == r;

            if nnz(currentROI) <= 0
                disp(['Warning: ROI ', int2str(r), ' not found. Skipped.'])
            end
            
            if nnz(currentROI) > 0
                roiOutline = find(currentROI);

                % we save time by limiting the number of pixels considered for each ROI (and neuropil)
                [roiXValues, roiYValues] = ind2sub([frameSize(1), frameSize(2)], roiOutline);

                roiMinX = ceil(min(roiXValues));
                roiMaxX = floor(max(roiXValues));
                roiMinY = ceil(min(roiYValues));
                roiMaxY = floor(max(roiYValues));
                
                currentROI = uint16(currentROI(roiMinX:roiMaxX, roiMinY:roiMaxY));
                
                if neuropilCorrection > 0
                    currentNeuropil = neuropilMasks{r} == 1;
                    
                    neuropilOutline = find(currentNeuropil);

                    [neuropilXValues, neuropilYValues] = ind2sub([frameSize(1), frameSize(2)], neuropilOutline);

                    neuropilMinX = ceil(min(neuropilXValues));
                    neuropilMaxX = floor(max(neuropilXValues));
                    neuropilMinY = ceil(min(neuropilYValues));
                    neuropilMaxY = floor(max(neuropilYValues));
                    
                    currentNeuropil = uint16(currentNeuropil(neuropilMinX:neuropilMaxX, neuropilMinY:neuropilMaxY));
                end

                for f = 1:numberOfFrames
                    
                    % extract ROI values as means
                    roiValues(f, r) = mean(mean(squeeze(frames(f, roiMinX:roiMaxX, roiMinY:roiMaxY)).*currentROI));

                    if neuropilCorrection > 0
                        neuropilValues(f, r) = mean(trimmean(squeeze(frames(f, neuropilMinX:neuropilMaxX, neuropilMinY:neuropilMaxY)).*currentNeuropil, 10));

                        correctedROIValues(f, r) = roiValues(f, r) - neuropilValues(f, r).*neuropilCorrection;
                        
                        % intensity should never go below 0
                        if correctedROIValues(f, r) <= 0
                            correctedROIValues(f, r) = 0.001;
                        end
                    end
                end      
            end
        end
        
        if GUIHandles.neuropilCorrection > 0
            csvwrite([GUIHandles.sbxPath, GUIHandles.sbxName, '.sig'], [correctedROIValues, neuropilValues, roiValues]);
        else
            csvwrite([GUIHandles.sbxPath, GUIHandles.sbxName, '.sig'], roiValues);
        end

        csvwrite([GUIHandles.sbxPath, GUIHandles.sbxName, '.bri'], GUIHandles.meanBrightness);
        
        GUIHandles.progressBarAxes.Visible = 'off';
        GUIHandles.progressBar.Visible = 'off';
            
        GUIHandles.progressBar.Data = [];

        GUIHandles.cancelExtractionButton.Visible = 'off';
        GUIHandles.extractFluorescencesButton.Visible = 'on';
        
        GUIHandles.statusText.String = 'Ready';
    
        guidata(GUI, GUIHandles);
    end
    
end

function cancelExtractionCallback(GUI, ~)

    GUIHandles = guidata(GUI);
        
    GUIHandles.progressBarAxes.Visible = 'off';
    GUIHandles.progressBar.Visible = 'off';

    GUIHandles.progressBar.Data = [];

    GUIHandles.cancelExtractionButton.Visible = 'off';
    GUIHandles.extractFluorescencesButton.Visible = 'on';

    GUIHandles.statusText.String = 'Ready';
    
    guidata(GUI, GUIHandles);

end

function undoCallback(GUI, ~)

    GUIHandles = guidata(GUI);
    
    if GUIHandles.selectedROI(1) == 0
        
        % undo (or redo) the last ROI mask edit by reverting to a previously saved version of it. note that this will not undo loading of a new ROI mask
        temp = GUIHandles.roiMask;
        GUIHandles.roiMask = GUIHandles.previousROIMask;
        GUIHandles.previousROIMask = temp;

        switch GUIHandles.undoButton.Visible
            case 'on'
                GUIHandles.undoButton.Visible = 'off';
                GUIHandles.redoButton.Visible = 'on';
            case 'off'
                GUIHandles.undoButton.Visible = 'on';
                GUIHandles.redoButton.Visible = 'off';
        end

        guidata(GUI, GUIHandles);

        updateROIImage(GUI);
    end
    
end

function deleteROIsCallback(GUI, ~)

    GUIHandles = guidata(GUI);
    
    % deletes the currently selected ROI
    if GUIHandles.selectedROI(1) > 0
        if ~strcmp(GUIHandles.manipulateType, 'Draw')
            GUIHandles.selectedROIImage.Visible = 'off';
            GUIHandles.neuropilImage.Visible = 'off';
            
            for r = length(GUIHandles.selectedROI):-1:1
                GUIHandles.previousROIMask(GUIHandles.selectedROIMask == GUIHandles.selectedROI(r)) = GUIHandles.selectedROI(r);
                
                % tick down the numbering of the ROIs
                GUIHandles.roiMask(GUIHandles.roiMask  > GUIHandles.selectedROI(r)) = GUIHandles.roiMask(GUIHandles.roiMask  > GUIHandles.selectedROI(r)) - 1;
            end
            
            GUIHandles.selectedROI = 0;
            
            GUIHandles.selectedROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
            GUIHandles.selectedROINeuropilMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
    
            % since this is a "new" edit, change redo button back to undo button if necessary
            if strcmp(GUIHandles.redoButton.Visible, 'on')
                GUIHandles.redoButton.Visible = 'off';
                GUIHandles.undoButton.Visible = 'on';
            end
    
            guidata(GUI, GUIHandles);

            updateROIImage(GUI);
    
            if strcmp(GUIHandles.manipulateType, 'Prune')
                updateHistogram(GUI);
            end
        end
    end
    
end

function saveROIsCallback(GUI, ~)
    
    GUIHandles = guidata(GUI);
    
    if GUIHandles.selectedROI(1) > 0
        selectedROIImageClickFunction(GUI);
    end
    
    GUIHandles = guidata(GUI);
    
    numberOfROIs = max(GUIHandles.roiMask(:));
    
    % create vertices from mask    
    vertices = cell(numberOfROIs, 1);
    
    for r = 1:numberOfROIs
        try
            vertices{r} = mask2poly(GUIHandles.roiMask == r, 'Inner', 'MinDist');
        catch
            vertices{r} = [];
        end
    end
    
    % change the name so it's correct in the .mat file
    roiMask = GUIHandles.roiMask;
        
    GUIHandles.statusText.String = 'Saving...';
    drawnow;
    
    if isfield(GUIHandles, 'roiMaskName')
        save([GUIHandles.roiMaskPath, GUIHandles.roiMaskName, '.segment'], 'roiMask', 'vertices');
    else
        save([GUIHandles.sbxPath, GUIHandles.sbxName, '.segment'], 'roiMask', 'vertices');
        
        GUIHandles.roiMaskPath = GUIHandles.sbxPath;
        GUIHandles.roiMaskName = GUIHandles.sbxName;
    end
        
    GUIHandles.statusText.String = 'Ready';
    
    guidata(GUI, GUIHandles);
    
end

function manipulateROIsButtonGroupSelectionFunction(GUI, eventdata)

    GUIHandles = guidata(GUI);
    
    % respond to changes in selected ROI manipulation type
    GUIHandles.manipulateType = get(eventdata.NewValue, 'String');
    
    switch GUIHandles.manipulateType
        case 'Draw'
            GUIHandles.drawShapeButtonGroup.Visible = 'on';
            
            GUIHandles.leftButton.Visible = 'off';
            GUIHandles.rightButton.Visible = 'off';
            GUIHandles.upButton.Visible = 'off';
            GUIHandles.downButton.Visible = 'off';
            
            GUIHandles.findROIButton.Visible = 'off';

            GUIHandles.fillROIButton.Visible = 'off';
            GUIHandles.clusterROIsButton.Visible = 'off';
            
            GUIHandles.histogramAxes.Visible = 'off';
            GUIHandles.histogramPlot.Visible = 'off';
        
            GUIHandles.predictFluorescenceButton.Visible = 'off';
            
            GUIHandles.islandSizeText.Visible = 'off';
            GUIHandles.islandSizeSlider.Visible = 'off';
    
            GUIHandles.viewNeuropilButton.Visible = 'off';
            GUIHandles.viewNeuropilButton.Value = 0;
            
            GUIHandles.neuropilSizeText.Visible = 'off';
            GUIHandles.neuropilSizeSlider.Visible = 'off';
            
            GUIHandles.neuropilImage.Visible = 'off';
            
            % reset the number of pixels to include in flood filling to make the first flood fill feel normal
            GUIHandles.floodPixels = GUIHandles.defaultFloodPixels;
        case 'Select'
            GUIHandles.drawShapeButtonGroup.Visible = 'off';
            
            GUIHandles.leftButton.Visible = 'on';
            GUIHandles.rightButton.Visible = 'on';
            GUIHandles.upButton.Visible = 'on';
            GUIHandles.downButton.Visible = 'on';
            
            GUIHandles.findROIButton.Visible = 'on';

            if length(GUIHandles.selectedROI) > 1
                GUIHandles.fillROIButton.Visible = 'off';
                GUIHandles.clusterROIsButton.Visible = 'on';
            else
                if GUIHandles.selectedROI > 1
                    GUIHandles.fillROIButton.Visible = 'on';
                end
                
                GUIHandles.clusterROIsButton.Visible = 'off';
            end
            
            GUIHandles.histogramAxes.Visible = 'off';
            GUIHandles.histogramPlot.Visible = 'off';
        
            GUIHandles.predictFluorescenceButton.Visible = 'off';
            
            GUIHandles.islandSizeText.Visible = 'off';
            GUIHandles.islandSizeSlider.Visible = 'off';
    
            GUIHandles.viewNeuropilButton.Visible = 'off';
            GUIHandles.viewNeuropilButton.Value = 0;
            
            GUIHandles.neuropilSizeText.Visible = 'off';
            GUIHandles.neuropilSizeSlider.Visible = 'off';
            
            GUIHandles.neuropilImage.Visible = 'off';
        case 'Prune'
            GUIHandles.drawShapeButtonGroup.Visible = 'off';
            
            GUIHandles.leftButton.Visible = 'off';
            GUIHandles.rightButton.Visible = 'off';
            GUIHandles.upButton.Visible = 'off';
            GUIHandles.downButton.Visible = 'off';
            
            GUIHandles.findROIButton.Visible = 'off';

            GUIHandles.fillROIButton.Visible = 'off';
            GUIHandles.clusterROIsButton.Visible = 'off';
            
            GUIHandles.histogramAxes.Visible = 'on';
            GUIHandles.histogramPlot.Visible = 'on';
        
            GUIHandles.predictFluorescenceButton.Visible = 'on';
            
            GUIHandles.islandSizeText.Visible = 'on';
            GUIHandles.islandSizeSlider.Visible = 'on';
    
            GUIHandles.viewNeuropilButton.Visible = 'on';
            
            % reset the pruning threshold to make the first scroll feel smooth
            GUIHandles.pruneThreshold = 0;
            
            % only prune one ROI at a time
            if length(GUIHandles.selectedROI) > 1
                for r = 1:length(GUIHandles.selectedROI) - 1
                    GUIHandles.roiMask(GUIHandles.selectedROIMask == GUIHandles.selectedROI(r)) = GUIHandles.selectedROI(r);
                    GUIHandles.selectedROIMask(GUIHandles.selectedROIMask == GUIHandles.selectedROI(r)) = 0;
                end
                
                GUIHandles.selectedROI = GUIHandles.selectedROI(end);
                
                guidata(GUI, GUIHandles);
                
                updateROIImage(GUI);
                updateSelectedROIImage(GUI);
            end
            
            if GUIHandles.selectedROI > 0
                GUIHandles.selectedROIPixelValues = GUIHandles.displayedImage.CData.*logical(GUIHandles.selectedROIImage.AlphaData);

                GUIHandles.previousROIMask = GUIHandles.roiMask;
                GUIHandles.allROIsMask = GUIHandles.roiMask > 0;
            
                % very important to make sure the selected ROI has been removed from the full ROI mask
                GUIHandles.roiMask(GUIHandles.roiMask == GUIHandles.selectedROI) = 0;
                
                guidata(GUI, GUIHandles);
            
                updateHistogram(GUI);
            end
    end
    
    guidata(GUI, GUIHandles);

end

function drawTypeButtonGroupSelectionFunction(GUI, eventdata)

    GUIHandles = guidata(GUI);
    
    % respond to changes in selected ROI drawing type
    GUIHandles.drawType = get(eventdata.NewValue, 'String');
    
    guidata(GUI, GUIHandles);

end

function moveCallback(GUI, eventdata)

    GUIHandles = guidata(GUI);
    
    if strcmp(GUIHandles.manipulateType, 'Select')        
        moveDirection = get(eventdata.Source, 'String');

        switch moveDirection
            case 'Left'
                shift = [-1, 2];
            case 'Right'
                shift = [1, 2];
            case 'Up'
                shift = [-1, 1];
            case 'Down'
                shift = [1, 1];
        end
        
        % if no ROI is selected, move entire mask
        if GUIHandles.selectedROI(1) == 0
            GUIHandles.roiMask = circshift(GUIHandles.roiMask, shift(1), shift(2));

            guidata(GUI,GUIHandles);

            updateROIImage(GUI);
        else
            GUIHandles.selectedROIMask = circshift(GUIHandles.selectedROIMask, shift(1), shift(2));

            guidata(GUI,GUIHandles);

            updateSelectedROIImage(GUI)
        end
    end

end

function fillROICallback(GUI,  ~)

    GUIHandles = guidata(GUI);

    if GUIHandles.selectedROI(1) > 0
        currentROI = GUIHandles.selectedROIMask;
        
        if nnz(currentROI) > 0                
            roiOutline = find(currentROI);

            [roiXValues, roiYValues] = ind2sub([GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)], roiOutline);

            roiMinX = ceil(min(roiXValues));
            roiMaxX = floor(max(roiXValues));
            roiMinY = ceil(min(roiYValues));
            roiMaxY = floor(max(roiYValues));

            currentROI = currentROI(roiMinX:roiMaxX, roiMinY:roiMaxY);
        end
    end
    
    % fill in pixels that are bordered by at least 5 pixels corresponding to the ROI
    for i = 2:size(currentROI, 1) - 1
        for j = 2:size(currentROI, 2) - 1
            if currentROI(i, j) == 0 && nnz(currentROI(i - 1:i + 1, j - 1:j + 1)) > 4
                currentROI(i, j) = GUIHandles.selectedROI(1);
            end
        end
    end
    
    GUIHandles.selectedROIMask(roiMinX:roiMaxX, roiMinY:roiMaxY) = currentROI;
    
    guidata(GUI, GUIHandles);
    
    updateSelectedROIImage(GUI);

end

function clusterROIsCallback(GUI,  ~)

    GUIHandles = guidata(GUI);
    
    GUIHandles.selectedROI = sort(GUIHandles.selectedROI);
    
    for r = 1:length(GUIHandles.selectedROI)
        GUIHandles.roiMask(GUIHandles.selectedROIMask == GUIHandles.selectedROI(r)) = GUIHandles.selectedROI(1);
    end
    
    % these loops are separated so they don't affect each other
    for r = length(GUIHandles.selectedROI):-1:2
        GUIHandles.roiMask(GUIHandles.roiMask  > GUIHandles.selectedROI(r)) = GUIHandles.roiMask(GUIHandles.roiMask  > GUIHandles.selectedROI(r)) - 1;
    end
    
    GUIHandles.selectedROI = GUIHandles.selectedROI(1);

    if strcmp(GUIHandles.undoButton.Visible, 'off')
        GUIHandles.redoButton.Visible = 'off';
        GUIHandles.undoButton.Visible = 'on';
    end

    GUIHandles.clusterROIsButton.Visible = 'off';
    
    guidata(GUI, GUIHandles);

end

function findROICallback(GUI, ~)

    GUIHandles = guidata(GUI);
    
    answer = inputdlg({'Find ROI:'}, 'Input for Subthreshold Analysis', 1);
    selectedROI = str2double(answer{1}) + 1;

    if length(selectedROI) > 1
        waitfor(msgbox('Error: Please search for one ROI at a time.'));
        error('Please search for one ROI at a time.');
    end
    
    % first, return any current selected ROI to the full ROI mask
    if GUIHandles.selectedROI ~= 0
        for r = 1:length(GUIHandles.selectedROI)        
            GUIHandles.roiMask(GUIHandles.selectedROIMask == GUIHandles.selectedROI(r)) = GUIHandles.selectedROI(r);
        end

        GUIHandles.selectedROIImage.Visible = 'off';

        GUIHandles.selectedROI = 0;

        GUIHandles.selectedROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));

        GUIHandles.fillROIButton.Visible = 'off';
        GUIHandles.clusterROIsButton.Visible = 'off';

        GUIHandles.deleteROIsButton.String = 'Delete ROI';

        guidata(GUI, GUIHandles);

        updateROIImage(GUI);
    end
    
    if any(GUIHandles.roiMask(:) == selectedROI);
        GUIHandles.selectedROI = selectedROI;
    
        if GUIHandles.selectedROI > 0
            GUIHandles.previousROIMask = GUIHandles.roiMask;
            GUIHandles.allROIsMask = GUIHandles.roiMask > 0;

            GUIHandles.selectedROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));

            % move the selected ROI from the normal ROI mask to the selected ROI mask
            GUIHandles.selectedROIMask(GUIHandles.roiMask == GUIHandles.selectedROI) = GUIHandles.selectedROI;

            GUIHandles.roiMask(GUIHandles.roiMask == GUIHandles.selectedROI) = 0;

            GUIHandles.selectedROINeuropilMask = (imdilate(GUIHandles.selectedROIMask, strel('disk', GUIHandles.neuropilSize)) - GUIHandles.allROIsMask) > 0;

            % reset this values so we don't start somewhere weird
            GUIHandles.minIslandSize = GUIHandles.defaultMinIslandSize;
            GUIHandles.islandSizeSlider.Value = GUIHandles.minIslandSize;

            GUIHandles.selectedROIImage.Visible = 'on';

            if strcmp(GUIHandles.manipulateType, 'Select')
                GUIHandles.fillROIButton.Visible = 'on';
            end

            guidata(GUI, GUIHandles);

            updateROIImage(GUI);
            updateSelectedROIImage(GUI);
        end
    end
    
end

function predictFluorescenceCallback(GUI, ~)

    GUIHandles = guidata(GUI);

    if GUIHandles.selectedROI(1) > 0
        currentROI = logical(GUIHandles.selectedROIImage.AlphaData);
        
        if nnz(currentROI) > 0        
            numberOfFrames = GUIHandles.Info.maxIndex + 1;

            roiValues = zeros(numberOfFrames, 1);
        
            roiOutline = find(currentROI);

            [roiXValues, roiYValues] = ind2sub([GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)], roiOutline);

            roiMinX = ceil(min(roiXValues));
            roiMaxX = floor(max(roiXValues));
            roiMinY = ceil(min(roiYValues));
            roiMaxY = floor(max(roiYValues));

            currentROI = uint16(currentROI(roiMinX:roiMaxX, roiMinY:roiMaxY));

            if GUIHandles.neuropilCorrection > 0            
                neuropilValues = zeros(numberOfFrames, 1);
                correctedROIValues = zeros(numberOfFrames, 1);

                currentNeuropil = GUIHandles.selectedROINeuropilMask == 1;
                
                neuropilOutline = find(currentNeuropil);

                [neuropilXValues, neuropilYValues] = ind2sub([GUIHandles.Info.sz(1), GUIHandles.Info.sz(2)], neuropilOutline);

                neuropilMinX = ceil(min(neuropilXValues));
                neuropilMaxX = floor(max(neuropilXValues));
                neuropilMinY = ceil(min(neuropilYValues));
                neuropilMaxY = floor(max(neuropilYValues));

                currentNeuropil = uint16(currentNeuropil(neuropilMinX:neuropilMaxX, neuropilMinY:neuropilMaxY));
            end
        
            GUIHandles.progressBarAxes.Visible = 'on';
            GUIHandles.progressBar.Visible = 'on';
            
            GUIHandles.predictFluorescenceButton.Visible = 'off';
            GUIHandles.cancelPredictionButton.Visible = 'on';
        
            GUIHandles.statusText.String = 'Predicting...';
            drawnow;

            for f = 1:numberOfFrames        
                frame = sbxRead(GUIHandles.Info, f - 1);

                roiValues(f) = mean(mean(frame(roiMinX:roiMaxX, roiMinY:roiMaxY).*currentROI));

                if GUIHandles.neuropilCorrection > 0
                    neuropilValues(f) = mean(trimmean(frame(neuropilMinX:neuropilMaxX, neuropilMinY:neuropilMaxY).*currentNeuropil, 10));
                end
            
                GUIHandles.progressBar.Data = linspace(0, round(f/numberOfFrames*100), round(f/numberOfFrames*100));
                drawnow;
            end
            
            if GUIHandles.neuropilCorrection > 0
                correctedROIValues = roiValues - neuropilValues*GUIHandles.neuropilCorrection;
                
                for i = 1:length(correctedROIValues)
                    if correctedROIValues(i) < 1
                        correctedROIValues(i) = 1;
                    end
                end
            end

            if isfield(GUIHandles.Info, 'timeStamps')
                frameTimes = GUIHandles.Info.timeStamps;
            else
                frameIndices = 0:GUIHandles.Info.maxIndex;
                frameTimes = (frameIndices*512)/(GUIHandles.Info.resfreq*(2 - GUIHandles.Info.scanmode));
            end

            figure('Name', 'Fluorescence Prediction', 'Units', 'normalized', 'Position', [0.125, 0.125, 0.75, 0.75], 'Color', 'w');

            fluorescenceAxes = axes();
            fluorescenceAxes.XLim = [0, frameTimes(end)];
            fluorescenceAxes.XLabel.String = 'Time [s]';
            fluorescenceAxes.YLabel.String = 'Fluorescence';

            if GUIHandles.neuropilCorrection > 0
                hold(fluorescenceAxes, 'on');

                plot(fluorescenceAxes, frameTimes, roiValues, 'Color', 'k');
                plot(fluorescenceAxes, frameTimes, correctedROIValues, 'Color', 'r');

                hold(fluorescenceAxes, 'off');            

                legend('Original', 'With Neuropil Correction');
                legend(fluorescenceAxes, 'boxoff');
            else
                plot(fluorescenceAxes, frameTimes, roiValues, 'Color', 'k');
            end                    
        
            GUIHandles.progressBarAxes.Visible = 'off';
            GUIHandles.progressBar.Visible = 'off';
            
            GUIHandles.progressBar.Data = [];
            
            GUIHandles.cancelPredictionButton.Visible = 'off';
            GUIHandles.predictFluorescenceButton.Visible = 'on';
        
            GUIHandles.statusText.String = 'Ready';
            
            guidata(GUI,GUIHandles);
        end
    end
    
end

function cancelPredictionCallback(GUI, ~)

    GUIHandles = guidata(GUI);
        
    GUIHandles.progressBarAxes.Visible = 'off';
    GUIHandles.progressBar.Visible = 'off';

    GUIHandles.progressBar.Data = [];

    GUIHandles.cancelPredictionButton.Visible = 'off'; 
    GUIHandles.predictFluorescenceButton.Visible = 'on';

    GUIHandles.statusText.String = 'Ready';
    
    guidata(GUI, GUIHandles);

end

function islandSizeSliderCallback(GUI, ~)
    
    GUIHandles = guidata(GUI);

    GUIHandles.minIslandSize = round(GUIHandles.islandSizeSlider.Value);
    
    guidata(GUI, GUIHandles);

    if GUIHandles.selectedROI(1) > 0
        pruneSelectedROI(GUI);
    end

end

function viewNeuropilButtonCallback(GUI, ~)

    GUIHandles = guidata(GUI);
    
    if GUIHandles.viewNeuropilButton.Value == 1
        if strcmp(GUIHandles.manipulateType, 'Prune')
            if GUIHandles.selectedROI(1) > 0
                GUIHandles.neuropilSizeText.Visible = 'on';
                GUIHandles.neuropilSizeSlider.Visible = 'on';
                
                GUIHandles.neuropilImage.Visible = 'on';
                
                GUIHandles.selectedROINeuropilMask = (imdilate(logical(GUIHandles.selectedROIImage.AlphaData), strel('disk', GUIHandles.neuropilSize)) - GUIHandles.allROIsMask) > 0;

                guidata(GUI, GUIHandles);

                updateNeuropilImage(GUI);
            end
        end
    else
        GUIHandles.neuropilSizeText.Visible = 'off';
        GUIHandles.neuropilSizeSlider.Visible = 'off';
        
        GUIHandles.neuropilImage.Visible = 'off';

        guidata(GUI, GUIHandles);
    end

end

function neuropilSizeSliderCallback(GUI, ~)
    
    GUIHandles = guidata(GUI);

    GUIHandles.neuropilSize = round(GUIHandles.neuropilSizeSlider.Value);        
    
    GUIHandles.selectedROINeuropilMask = (imdilate(logical(GUIHandles.selectedROIImage.AlphaData), strel('disk', GUIHandles.neuropilSize)) - GUIHandles.allROIsMask) > 0;

    guidata(GUI, GUIHandles);

    if GUIHandles.viewNeuropilButton.Value == 1
        updateNeuropilImage(GUI)
    end

end

function brightnessSliderCallback(GUI, ~)
    
    GUIHandles = guidata(GUI);

    GUIHandles.brightness = GUIHandles.brightnessSlider.Value;
    
    guidata(GUI, GUIHandles);

    updateImageDisplay(GUI)

end

function contrastSliderCallback(GUI, ~)
    
    GUIHandles = guidata(GUI);

    GUIHandles.contrast = GUIHandles.contrastSlider.Value;        
    
    guidata(GUI, GUIHandles);
    
    updateImageDisplay(GUI)

end

function mouseOverFunction(GUI, ~)

    GUIHandles = guidata(GUI);

    % if the local cross-correlation option is selected, move the little square around with the mouse
    if GUIHandles.ccLocalButton.Value == 1
        updateImageDisplay(GUI);
    end
    
end

function scrollFunction(GUI, eventdata)

    GUIHandles = guidata(GUI);
    
    if strcmp(get(GUI, 'CurrentModifier'), 'shift')
        modifier = 0.1;
    else
        modifier = 1;
    end
    
    if strcmp(get(GUI, 'CurrentModifier'), 'control')          
        if ~isempty(GUIHandles.zoomCenter)
            GUIHandles.windowSize = min(GUIHandles.maxWindowSize, max(1, GUIHandles.windowSize + (eventdata.VerticalScrollAmount*eventdata.VerticalScrollCount)*15/3));

            XLim = [GUIHandles.zoomCenter(1) - GUIHandles.windowSize, GUIHandles.zoomCenter(1) + GUIHandles.windowSize];
            YLim = [GUIHandles.zoomCenter(2) - GUIHandles.windowSize*GUIHandles.windowRatio, GUIHandles.zoomCenter(2) + GUIHandles.windowSize*GUIHandles.windowRatio];

            if XLim(1) < GUIHandles.XLim(1)
                XLim(1) = GUIHandles.XLim(1);
                XLim(2) = GUIHandles.XLim(1) + 2*GUIHandles.windowSize;
            elseif XLim(2) > GUIHandles.XLim(2)
                XLim(1) = GUIHandles.XLim(2) - 2*GUIHandles.windowSize;
                XLim(2) = GUIHandles.XLim(2);
            end
            if YLim(1) < GUIHandles.YLim(1)
                YLim(1) = GUIHandles.YLim(1);
                YLim(2) = GUIHandles.YLim(1) + 2*GUIHandles.windowSize*GUIHandles.windowRatio;
            elseif YLim(2) > GUIHandles.YLim(2)
                YLim(1) = GUIHandles.YLim(2) - 2*GUIHandles.windowSize*GUIHandles.windowRatio;
                YLim(2) = GUIHandles.YLim(2);
            end

            GUIHandles.imageAxes.XLim = XLim;
            GUIHandles.imageAxes.YLim = YLim;

            guidata(GUI, GUIHandles);
        end
    elseif strcmp(GUIHandles.selectedROIImage.Visible, 'on') || strcmp(GUIHandles.candidateROIImage.Visible, 'on')
        switch GUIHandles.manipulateType
            case 'Draw'

                % if drawing, scrolling affects the number of pixels added via flood-filling
                GUIHandles.floodPixels = max(10, GUIHandles.floodPixels - (eventdata.VerticalScrollAmount*eventdata.VerticalScrollCount)*10/3*modifier);
                
                guidata(GUI, GUIHandles);

                updateCandidateROIImage(GUI);
            case 'Select'
                if strcmp(GUIHandles.imageType, 'Rolling Average')
                    GUIHandles.frameSlider.Value = min(GUIHandles.Info.maxIndex, max(0, GUIHandles.frameSlider.Value - (eventdata.VerticalScrollAmount*eventdata.VerticalScrollCount)*50/3*modifier));
                    
                    guidata(GUI, GUIHandles);

                    updateImageDisplay(GUI);
                end
            case 'Prune'

                % if pruning, scrolling changes the intensity threshold below which pixels are discarded from the selected ROI
                if GUIHandles.selectedROI(1) > 0
                    GUIHandles.pruneThreshold = min(1, max(0, GUIHandles.pruneThreshold + (eventdata.VerticalScrollAmount*eventdata.VerticalScrollCount)*0.001*modifier));
                    
                    guidata(GUI, GUIHandles);

                    pruneSelectedROI(GUI);
                end
        end
    else
        if strcmp(GUIHandles.imageType, 'Rolling Average')
            GUIHandles.frameSlider.Value = min(GUIHandles.Info.maxIndex, max(0, GUIHandles.frameSlider.Value - (eventdata.VerticalScrollAmount*eventdata.VerticalScrollCount)*50/3*modifier));
        
            guidata(GUI, GUIHandles);

            updateImageDisplay(GUI);
        end
    end
    
end

function keyPressFunction(GUI, eventdata)
    
    switch eventdata.Key
        case 'u'
            undoCallback(GUI);
        case 'r'
            undoCallback(GUI);
        case 'd'
            deleteROIsCallback(GUI);
        case 'control'
            GUIHandles = guidata(GUI);
            
            if isempty(GUIHandles.zoomCenter)
                zoomCenter = get(GUIHandles.imageAxes, 'CurrentPoint');
                GUIHandles.zoomCenter = zoomCenter(1, 1:2);

                guidata(GUI, GUIHandles);
            end
    end
    
end

function keyReleaseFunction(GUI, eventdata)
    
    switch eventdata.Key
        case 'control'
            GUIHandles = guidata(GUI);
            
            GUIHandles.zoomCenter = [];
            
            guidata(GUI, GUIHandles);
    end
    
end

function roiImageClickFunction(GUI, ~)

    GUIHandles = guidata(GUI);
    
    % if draw ROI option is selected, draw ROI outlines using specified method
    if strcmp(GUIHandles.manipulateType, 'Draw')
        if strcmp(GUIHandles.drawType, 'Pixel')
            currentPoint = get(GUIHandles.imageAxes, 'CurrentPoint');
            currentPoint = round(currentPoint(1, 1:2)');

            if (GUIHandles.Info.sz(2) > currentPoint(1)) && (currentPoint(1) > 0) && (GUIHandles.Info.sz(1) > currentPoint(2)) && (currentPoint(2) > 0)
                GUIHandles.imageToBeFlooded = GUIHandles.displayedImage.CData;
                
                GUIHandles.floodCenter = currentPoint;

                GUIHandles.candidateROIImage.Visible = 'on';
                
                guidata(GUI, GUIHandles);

                computeROIFlood(GUI);

                updateCandidateROIImage(GUI);
            end
        else
            switch GUIHandles.drawType
                case 'Free Hand'
                    shape = imfreehand;
                case 'Polygon'
                    shape = impoly;
                case 'Ellipse'
                    shape = imellipse;
                case 'Rectangle'
                    shape = imrect;
            end

            wait(shape);

            GUIHandles.previousROIMask = GUIHandles.roiMask;
            
            GUIHandles.imageToBeFlooded = GUIHandles.displayedImage.CData.*shape.createMask(GUIHandles.displayedImage);
            
            delete(shape);
            
            [~, index] = max(GUIHandles.imageToBeFlooded(:));
            [row, column] = ind2sub(size(GUIHandles.imageToBeFlooded), index);
            GUIHandles.floodCenter = [column, row];

            GUIHandles.candidateROIImage.Visible = 'on';
            
            guidata(GUI, GUIHandles);
            
            computeROIFlood(GUI);
            
            updateCandidateROIImage(GUI);
        end
    else
        currentPoint = get(GUIHandles.imageAxes, 'CurrentPoint');
        currentPoint = round(currentPoint(1, 1:2)');  

        if (GUIHandles.Info.sz(2) > currentPoint(1)) && (currentPoint(1) > 0) && (GUIHandles.Info.sz(1) > currentPoint(2)) && (currentPoint(2) > 0)
            GUIHandles.selectedROI = GUIHandles.roiMask(currentPoint(2), currentPoint(1));

            if GUIHandles.selectedROI > 0
                GUIHandles.previousROIMask = GUIHandles.roiMask;
                GUIHandles.allROIsMask = GUIHandles.roiMask > 0;
                
                GUIHandles.selectedROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
            
                % move the selected ROI from the normal ROI mask to the selected ROI mask
                for r = 1:length(GUIHandles.selectedROI)
                    GUIHandles.selectedROIMask(GUIHandles.roiMask == GUIHandles.selectedROI(r)) = GUIHandles.selectedROI(r);
                end

                for r = 1:length(GUIHandles.selectedROI)
                    GUIHandles.roiMask(GUIHandles.roiMask == GUIHandles.selectedROI(r)) = 0;
                end
                    
                GUIHandles.selectedROINeuropilMask = (imdilate(GUIHandles.selectedROIMask, strel('disk', GUIHandles.neuropilSize)) - GUIHandles.allROIsMask) > 0;
                
                % reset this values so we don't start somewhere weird
                GUIHandles.minIslandSize = GUIHandles.defaultMinIslandSize;
                GUIHandles.islandSizeSlider.Value = GUIHandles.minIslandSize;
                
                GUIHandles.selectedROIImage.Visible = 'on';
        
                if strcmp(GUIHandles.manipulateType, 'Select')
                    GUIHandles.fillROIButton.Visible = 'on';
                end

                guidata(GUI, GUIHandles);

                updateROIImage(GUI);
                updateSelectedROIImage(GUI);

                % if pruning, calculate the pixel values of the selected ROI
                if strcmp(GUIHandles.manipulateType, 'Prune')
                    GUIHandles.selectedROIPixelValues = GUIHandles.displayedImage.CData.*logical(GUIHandles.selectedROIImage.AlphaData);
                    
                    guidata(GUI, GUIHandles);
                    
                    updateHistogram(GUI);
                end
                
                if GUIHandles.viewNeuropilButton.Value == 1
                    GUIHandles.neuropilSizeText.Visible = 'on';
                    GUIHandles.neuropilSizeSlider.Visible = 'on';
        
                    GUIHandles.neuropilImage.Visible = 'on';
                    
                    guidata(GUI, GUIHandles);
                    
                    updateNeuropilImage(GUI);
                end
            end
        end
    end
    
end

function candidateROIImageClickFunction(GUI, ~)

    GUIHandles = guidata(GUI);
    
    % save the drawn ROI upon second click
    newROIMask = GUIHandles.floodOrder < GUIHandles.floodPixels;
    newROIMask(GUIHandles.roiMask > 0) = 0;

    GUIHandles.previousROIMask = GUIHandles.roiMask;
            
    GUIHandles.roiMask = GUIHandles.roiMask + (max(GUIHandles.roiMask(:)) + 1)*newROIMask;
    
    GUIHandles.candidateROIImage.Visible = 'off';
    
    if strcmp(GUIHandles.undoButton.Visible, 'off')
        GUIHandles.redoButton.Visible = 'off';
        GUIHandles.undoButton.Visible = 'on';
    end
    
    guidata(GUI, GUIHandles);
    
    updateROIImage(GUI);
    
end
    
function selectedROIImageClickFunction(GUI, ~)

    GUIHandles = guidata(GUI);
    
    if strcmp(GUIHandles.manipulateType, 'Select')
        if strcmp(get(GUIHandles.parentFigure, 'CurrentModifier'), 'shift')
            currentPoint = get(GUIHandles.imageAxes, 'CurrentPoint');
            currentPoint = round(currentPoint(1, 1:2)');  

            if (GUIHandles.Info.sz(2) > currentPoint(1)) && (currentPoint(1) > 0) && (GUIHandles.Info.sz(1) > currentPoint(2)) && (currentPoint(2) > 0)
                GUIHandles.selectedROI(end + 1) = GUIHandles.roiMask(currentPoint(2), currentPoint(1));
                
                if GUIHandles.selectedROI(end) > 0
            
                    % move the selected ROI from the normal ROI mask to the selected ROI mask
                    GUIHandles.selectedROIMask(GUIHandles.roiMask == GUIHandles.selectedROI(end)) = GUIHandles.selectedROI(end);

                    GUIHandles.roiMask(GUIHandles.roiMask == GUIHandles.selectedROI(end)) = 0;
                    
                    GUIHandles.selectedROINeuropilMask = (imdilate(GUIHandles.selectedROIMask, strel('disk', GUIHandles.neuropilSize)) - GUIHandles.allROIsMask) > 0;
                    
                    GUIHandles.fillROIButton.Visible = 'off';
                    GUIHandles.clusterROIsButton.Visible = 'on';
                    
                    GUIHandles.deleteROIsButton.String = 'Delete ROIs';
                    
                    guidata(GUI, GUIHandles);

                    updateROIImage(GUI);
                    updateSelectedROIImage(GUI);
                end
            end
            
            selectedAdditionalROI = 1;
        else
            % or return selected ROI to the normal ROI mask    
            for r = 1:length(GUIHandles.selectedROI)        
                GUIHandles.roiMask(GUIHandles.selectedROIMask == GUIHandles.selectedROI(r)) = GUIHandles.selectedROI(r);
            end
            
            selectedAdditionalROI = 0;
        end
    elseif strcmp(GUIHandles.manipulateType, 'Prune')
        GUIHandles.islandImage.Visible = 'off';
        GUIHandles.islandMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
        
        % save pruned ROI    
        GUIHandles.roiMask = GUIHandles.roiMask + GUIHandles.selectedROI*logical(GUIHandles.selectedROIImage.AlphaData);
        
        if nnz(GUIHandles.selectedROIImage.AlphaData) == 0
            GUIHandles.roiMask(GUIHandles.roiMask  > GUIHandles.selectedROI) = GUIHandles.roiMask(GUIHandles.roiMask  > GUIHandles.selectedROI) - 1;
        end

        if strcmp(GUIHandles.undoButton.Visible, 'off')
            GUIHandles.redoButton.Visible = 'off';
            GUIHandles.undoButton.Visible = 'on';
        end
            
        selectedAdditionalROI = 0;
    else
        for r = 1:length(GUIHandles.selectedROI)
            GUIHandles.roiMask(GUIHandles.selectedROIMask == GUIHandles.selectedROI(r)) = GUIHandles.selectedROI(r);
        end
            
        selectedAdditionalROI = 0;
    end
    
    if selectedAdditionalROI == 0
        GUIHandles.islandImage.Visible = 'off';
        GUIHandles.selectedROIImage.Visible = 'off';

        if GUIHandles.viewNeuropilButton.Value == 1
            GUIHandles.neuropilSizeText.Visible = 'off';
            GUIHandles.neuropilSizeSlider.Visible = 'off';

            GUIHandles.neuropilImage.Visible = 'off';
        end

        GUIHandles.selectedROI = 0;

        GUIHandles.selectedROIMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));

        GUIHandles.fillROIButton.Visible = 'off';
        GUIHandles.clusterROIsButton.Visible = 'off';
        
        GUIHandles.deleteROIsButton.String = 'Delete ROI';

        guidata(GUI, GUIHandles);

        if strcmp(GUIHandles.manipulateType, 'Prune')
            updateHistogram(GUI);
        end

        updateROIImage(GUI);
    end
    
end

function updateImageDisplay(GUI)
    
    GUIHandles = guidata(GUI);
    
    % don't try to leave this static, since a fresh image needs to be fed to overlayCrossCorrelation 
    switch GUIHandles.imageType
        case 'Mean'            
            GUIHandles.displayedImage.CData = GUIHandles.meanReference;
        case 'Max Intensity'            
            GUIHandles.displayedImage.CData = GUIHandles.maxIntensityProjection;
        case 'Cross-Correlation'            
            GUIHandles.displayedImage.CData = GUIHandles.ccImage;
        case 'PCA'            
            GUIHandles.displayedImage.CData = GUIHandles.pcaImage;
        case 'Rolling Average'
            index = round(GUIHandles.frameSlider.Value);
            
            frames = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2), GUIHandles.rollingWindowSize*2 + 1);
            
            if index - GUIHandles.rollingWindowSize < 0
                start = 0;
                finish = GUIHandles.rollingWindowSize*2;
            elseif index + GUIHandles.rollingWindowSize > GUIHandles.Info.maxIndex
                start = GUIHandles.Info.maxIndex - GUIHandles.rollingWindowSize*2;
                finish = GUIHandles.Info.maxIndex;
            else
                start = index - GUIHandles.rollingWindowSize;
                finish = index + GUIHandles.rollingWindowSize;
            end
                
            for i = 0:finish - start
                frames(:, :, i  + 1) = sbxRead(GUIHandles.Info, start + i);
            end
            
            rollingAverage = double(mean(frames, 3))/double(intmax('uint16'));
            
            % unity normalize each frame to intmax('uint16') to use the same color-scale as the pre-computed values
            GUIHandles.displayedImage.CData = rollingAverage;
        case 'Frame-by-Frame'
            index = round(GUIHandles.frameSlider.Value);
            
            frame = sbxRead(GUIHandles.Info, index);
            
            % unity normalize each frame to intmax('uint16') to use the same color-scale as the pre-computed values
            GUIHandles.displayedImage.CData = double(frame)/double(intmax('uint16'));
    end
    
    if GUIHandles.brightness ~= 0.0
        GUIHandles.displayedImage.CData = GUIHandles.displayedImage.CData + GUIHandles.brightness;
    end
    
    if GUIHandles.contrast ~= 1.0
        maximum = max(GUIHandles.displayedImage.CData(:));
        minimum = min(GUIHandles.displayedImage.CData(:));
        
        range = (maximum - minimum)/GUIHandles.contrast;
        
        GUIHandles.displayedImage.CData = (GUIHandles.displayedImage.CData - minimum)/range + minimum;
    end
        
    % if selected, show the local cross-correlation
    if GUIHandles.ccLocalButton.Value == 1
        guidata(GUI, GUIHandles);
        
        overlayCrossCorrelation(GUI);
    end
        
    guidata(GUI, GUIHandles);
    
end

function updateROIImage(GUI)

    GUIHandles = guidata(GUI);

    GUIHandles.roiImage.AlphaData = 0.5*(GUIHandles.roiMask > 0);
        
    guidata(GUI, GUIHandles);
    
end
    
function updateSelectedROIImage(GUI)

    GUIHandles = guidata(GUI);

    GUIHandles.selectedROIImage.AlphaData = 0.5*(GUIHandles.selectedROIMask > 0);

    guidata(GUI, GUIHandles);
    
end

function updateHistogram(GUI)

    GUIHandles = guidata(GUI);

    % update the pixel intensity histogram
    if GUIHandles.selectedROI > 0
        temp = GUIHandles.displayedImage.CData.*logical(GUIHandles.selectedROIImage.AlphaData);
    else
        temp = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));
    end

    GUIHandles.histogramPlot.Data = nonzeros(temp(:));

    if max(GUIHandles.histogramPlot.Values) > 0
        GUIHandles.histogramAxes.YLim = [0, max(GUIHandles.histogramPlot.Values)];
    else
        GUIHandles.histogramAxes.YLim = [0, 1];
    end

    % rescale the histogram's bins
    GUIHandles.histogramPlot.BinMethod = 'auto';

    guidata(GUI, GUIHandles);

end
    
function updateNeuropilImage(GUI)

    GUIHandles = guidata(GUI);

    GUIHandles.neuropilImage.AlphaData = 0.5*(GUIHandles.selectedROINeuropilMask > 0);

    guidata(GUI, GUIHandles);
    
end
    
function updateCandidateROIImage(GUI)

    GUIHandles = guidata(GUI);

    % adjust the flood-filling of the candidate ROI   
    GUIHandles.candidateROIImage.AlphaData = 0.5*(GUIHandles.floodOrder < GUIHandles.floodPixels);

    guidata(GUI, GUIHandles);
    
end
    
function pruneSelectedROI(GUI)

    GUIHandles = guidata(GUI);
    
    prunedROIMask = GUIHandles.selectedROIPixelValues > GUIHandles.pruneThreshold;
    
    if GUIHandles.minIslandSize > 0
        
        % this mask is for clusters of pixels that will be removed due to small size
        islandMask = zeros(GUIHandles.Info.sz(1), GUIHandles.Info.sz(2));

        prunedROIProperties = regionprops(prunedROIMask, 'Area', 'PixelList');

        for r = 1:length(prunedROIProperties)
            if prunedROIProperties(r).Area < GUIHandles.minIslandSize
                for i = 1:size(prunedROIProperties(r).PixelList, 1)
                    column = prunedROIProperties(r).PixelList(i, 1);
                    row = prunedROIProperties(r).PixelList(i, 2);

                    islandMask(row, column) = prunedROIMask(row, column);
                    prunedROIMask(row, column) = 0;
                end
            end
        end

        GUIHandles.islandImage.AlphaData = 0.5*(islandMask);
    
        GUIHandles.islandImage.Visible = 'on';
    end
    
    GUIHandles.selectedROIImage.AlphaData = 0.5*(prunedROIMask);        
    
    % don't forget to update the neuropil as ROI is pruned
    GUIHandles.selectedROINeuropilMask = (imdilate(logical(GUIHandles.selectedROIImage.AlphaData), strel('disk', GUIHandles.neuropilSize)) - GUIHandles.allROIsMask) > 0;
        
    guidata(GUI, GUIHandles);
    
    if GUIHandles.viewNeuropilButton.Value == 1
        updateNeuropilImage(GUI);
    end
    
    if strcmp(GUIHandles.manipulateType, 'Prune')
        updateHistogram(GUI);
    end
    
end
   
function computeROIFlood(GUI)

    GUIHandles = guidata(GUI);
    
    floodCenter = round(GUIHandles.floodCenter(2:-1:1));
    
    % set the maximum region to grow
    if strcmp(GUIHandles.drawType, 'Pixel')
        maxRegionSize = min(nnz(GUIHandles.imageToBeFlooded), max(250, 5*GUIHandles.floodPixels));
    else
        maxRegionSize = nnz(GUIHandles.imageToBeFlooded);
    end
        
    [~, floodOrder] = regionGrowing(GUIHandles.imageToBeFlooded, [floodCenter(1), floodCenter(2)], maxRegionSize);
    
    GUIHandles.floodOrder = floodOrder;
    
    guidata(GUI, GUIHandles);
    
end
    
function overlayCrossCorrelation(GUI)

    GUIHandles = guidata(GUI);

    ccLocalSize = size(GUIHandles.ccLocal, 3);
    
    cursor = get(GUIHandles.imageAxes, 'CurrentPoint');
    cursor = round(cursor(:, 1:2)');
    
    cursor = [592, 592; 98, 98];
    
    if (GUIHandles.Info.sz(2) - ccLocalSize/2 > cursor(1)) && (cursor(1) > ccLocalSize/2) && (GUIHandles.Info.sz(1) - ccLocalSize/2 > cursor(2)) && (cursor(2) > ccLocalSize/2)

        % distance in pixels of the selected coordinate from 0/0
        dx = -floor(cursor(2:-1:1)) + ccLocalSize;
        dx(1) = dx(1) - 1;
        dx(2) = dx(2) - 1;

        % shift window we want to paste into so that it starts at 0/0
        overlaidImage = circshift(GUIHandles.displayedImage.CData, dx);

        % pull out data from precomputed local cross-correlation matrix
        R = squeeze(GUIHandles.ccLocal(max(floor(cursor(2)), 1), max(floor(cursor(1)), 1), :, :));

        % double resolution and convolve image with a .5, 1, .5 kernel
        R2 = conv2([.5, 1, .5], [.5, 1, .5], R, 'same');
        A = R(2:end - 1, 2:end - 1);

        R(ceil(end/2), ceil(end/2)) = 0;
        rg = 1:size(A, 1);

        overlaidImage(rg + floor(ccLocalSize/2), rg + floor(ccLocalSize/2)) = A/(max(R(:)) + .01);

        % shift window back to its original position
        GUIHandles.displayedImage.CData = circshift(overlaidImage, -dx);
    end
    
end

function [floodedImage, floodOrder] = regionGrowing(originalImage, seedpoint, maxRegionSize)

    %REGIONGROWING Grow a region in an image from a specified seedpoint.
    %   [floodedImage, floodOrder] = REGIONGROWING(originalImage, seedpoint, maxRegionSize) iteratively grows a region around the seedpoint by comparing unallocated neighboring pixels to the existing region. The pixel with the smallest difference between its intensity value and the mean of the existing region is absorbed into the region. Growing ends when the size of the flooded region exceeds the specified limit.
    % 
    %   floodedImage: logical
    %       Logical matrix containing image of region.
    %
    %   floodOrder: matrix
    %       Matrix containing the order of pixels added to floodedImage.
    %
    %   originalImage: image
    %       Can be any class.
    %
    %   seedpoint: double
    %       Coordinates of seedpoint.
    %
    %   maxRegionSize: double
    %       Upper limit on pixels allowed into flooded image.
    %
    %   Original Author: D. Kroon, University of Twente
    
    floodedImage = zeros(size(originalImage));
    
    % free memory to store neighbors of the (segmented) region
    freeNeighbors = 10000; 
    neighborPositions = 0;
    neighborList = zeros(freeNeighbors, 3); 

    % neighbor locations (footprint)
    neighbors = [-1, 0; 1, 0; 0, -1; 0, 1];
    
    % number of pixels in region
    regionSize = 0; 

    % mean pixel intensity in region
    regionMean = originalImage(seedpoint(1), seedpoint(2));
    
    x = seedpoint(1);
    y = seedpoint(2);
    
    floodOrder = inf(size(originalImage));
    floodOrder(x, y) = 1;
    
    % start region growing until max region size is achieved
    while regionSize <= maxRegionSize
        
        % add the latest approved pixel
        floodedImage(x, y) = 2; 
        
        regionSize = regionSize + 1;
        
        floodOrder(x, y) = regionSize;
        
        % add new neighbors pixels
        for j = 1:4

            % calculate the neighbor coordinate
            xNeighbor = x + neighbors(j, 1); 
            yNeighbor = y + neighbors(j, 2);

            % check if neighbor is inside or outside the image
            inside = (xNeighbor >= 1) && (yNeighbor >= 1) && (xNeighbor <= size(originalImage, 1)) && (yNeighbor <= size(originalImage, 2));

            % add neighbor if inside and not already part of the segmented area
            if inside && (floodedImage(xNeighbor, yNeighbor) == 0)
                neighborPositions = neighborPositions + 1;
                neighborList(neighborPositions, :) = [xNeighbor, yNeighbor, originalImage(xNeighbor, yNeighbor)]; 
                floodedImage(xNeighbor, yNeighbor) = 1;
            end
        end

        % add a new block of free memory
        if neighborPositions + 10 > freeNeighbors
            freeNeighbors = freeNeighbors + 10000; 
            neighborList(neighborPositions + 1:freeNeighbors, :) = 0; 
        end

        % find neighbor pixel with highest intensity
        [~, index] = max(neighborList(1:neighborPositions, 3));
        
        % save the x and y coordinates of the pixel (for the neighbor add proccess)
        x = neighborList(index, 1); 
        y = neighborList(index, 2);

        % calculate the new mean of the region
        regionMean = (regionMean*regionSize + neighborList(index, 3))/(regionSize + 1);

        % remove the pixel from the neighbor (check) list
        neighborList(index, :) = neighborList(neighborPositions, :); 
        neighborPositions = neighborPositions - 1;
    end

    % return the segmented area as a logical matrix
    floodedImage = floodedImage == 2;
    
end