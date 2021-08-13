function [phaseDifferences, rowShifts, columnShifts, result] = sbxRigid(Info, Parameters)

    %SBXRIGID Motion-correct rigid imaging data contained in an .sbx file.
    %   [result, phaseDifferences, rowShifts, columnShifts] = SBXRIGID(Info, Parameters) uses a rigid algorithm to register imaging data frame-by-frame and saves to a new .sbx file.
    %
    %   Info: structure 
    %       Info structure generated by sbxInfo from corresponding .mat file.
    %
    %   Parameters: structure
    %       Optional input containing parameter specifications.
    %
    %   phaseDifferences: array
    %       Contains frame-by-frame phase differences calculated by dftregistration.
    %
    %   rowShifts: array
    %       Contains frame-by-frame row shifts calculated by dftregistration.
    %
    %   columnShifts: array
    %       Contains frame-by-frame column shifts calculated by dftregistration.
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

    if exist([Info.Directory.folder, Info.Directory.name, '.rigid'], 'file')
        waitfor(msgbox('Error: Data has already been rigidly aligned.'));
        error('Data has already been rigidly aligned.');
    elseif exist([Info.Directory.folder, Info.Directory.name, '.align'], 'file')
        waitfor(msgbox('Error: Data has already been rigidly aligned.'));
        error('Data has already been rigidly aligned.');
    end
    
    if ~exist('Parameters', 'var')
        GUI = false;
        
        % set the pixel intensity value to use as a threshold for selecting reference frames
        threshold = 0;
        
        % set sigma of Gaussian filter applied to image to smooth out pixel artefacts - 0 is no filtering
        gaussianFilter = 0.0;
        
        % set the number of consecutive times to motion correct
        passes = 3;
        
        % set number of (random) frames to use in generating a reference image
        sampleSize = 1000;
        
        % specify cropping in pixels: [from left, from right, from top, from bottom]
        frameCrop = [0, 0, 0, 0];
        
        % specify upsampling factor
        subpixelFactor = 1;
    else
        if ~isfield(Parameters, 'GUI')
            GUI = false;
        else
            GUI = Parameters.GUI;
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
        if ~isfield(Parameters, 'passes')
            passes = 3;
        else
            passes = Parameters.passes;
        end
        if ~isfield(Parameters, 'sampleSize')
            sampleSize = 1000;
        else
            sampleSize = Parameters.sampleSize;
        end
        if ~isfield(Parameters, 'frameCrop')
            frameCrop = [0, 0, 0, 0];
        else
            frameCrop = Parameters.frameCrop;
        end
        if ~isfield(Parameters, 'subpixelFactor')
            subpixelFactor = 1;
        else
            subpixelFactor = Parameters.subpixelFactor;
        end
    end
    
    % the first few frames always suck while the galvo winds up - exclude them
    offset = 10;
    
    if sampleSize > Info.maxIndex + 1 - offset
        sampleSize = Info.maxIndex + 1 - offset;
    end
    
    if GUI
        progressBar = waitbar(0, 'Generating reference images...', 'Name', [Info.Directory.name, ': sbxRigid'], 'CreateCancelBtn', 'setappdata(gcbf, ''Canceling'', 1)');
        setappdata(progressBar, 'Canceling', 0);
    end
    
    if any(frameCrop > 0)
        reference = zeros(Info.sz(1) - frameCrop(3) - frameCrop(4), Info.sz(2) - frameCrop(1) - frameCrop(2));
    else
        reference = zeros(Info.sz);
    end
    
    for pass = 1:passes + 1
        framePool = 0:Info.maxIndex;
        
        temp = zeros(size(reference));
        
        f = 1;
        
        % first generate the references
        while f <= sampleSize
            index = randi(Info.maxIndex + 1 - offset) - 1 + offset;
            
            if ismember(index, framePool)
                frame = sbxRead(Info, index);
                
                if max(frame(:)) > threshold
                    
                    % remove frame from frame pool to keep each index unique
                    framePool(framePool == index) = [];
                    
                    if any(frameCrop > 0)
                        frame = frame(frameCrop(3) + 1:Info.sz(1) - frameCrop(4), frameCrop(1) + 1:Info.sz(2) - frameCrop(2));
                    end

                    if gaussianFilter > 0
                        frame = imgaussfilt(frame, gaussianFilter);
                    end

                    if pass > 1
                        [~, registeredFrame] = dftregistration(fft2(reference), fft2(frame), subpixelFactor);

                        registeredFrame = abs(ifft2(registeredFrame));

                        % adjust values just in case
                        originalMinimum = double(min(frame(:)));
                        originalMaximum = double(max(frame(:)));
                        registeredMinimum = min(registeredFrame(:));
                        registeredMaximum = max(registeredFrame(:));

                        frame = (registeredFrame - registeredMinimum)/(registeredMaximum - registeredMinimum)*(originalMaximum - originalMinimum) + originalMinimum;
                    end

                    temp = max(temp, double(frame));
                    
                    f = f + 1;

                    if GUI
                        if getappdata(progressBar, 'Canceling')
                            delete(progressBar);

                            phaseDifferences = 0;
                            rowShifts = 0;
                            columnShifts = 0;

                            result = 'Canceled';
                            return
                        end
                    end
                else
                
                    % if frame doesn't have anything above threshold, then remove it from the framePool
                    framePool(framePool == index) = [];
                end
            end
            
            if isempty(framePool)
                warning(['Only ', int2str(f - 1), ' frames exceeded pixel intensity threshold.']);
                break
            end
        end
        
        reference = uint16(temp);
        
        imwrite(reference, [Info.Directory.folder, Info.Directory.name, '_reference_', int2str(pass), '.png'], 'bitdepth', 16);
            
        if GUI
            if getappdata(progressBar, 'Canceling')
                delete(progressBar);

                phaseDifferences = 0;
                rowShifts = 0;
                columnShifts = 0;
                
                result = 'Canceled';
                return
            else
                waitbar(pass/(passes + 1), progressBar);
            end
        end
    
        % on the last pass, register every frame
        if pass == passes + 1
            if GUI
                if getappdata(progressBar, 'Canceling')
                    delete(progressBar);

                    phaseDifferences = 0;
                    rowShifts = 0;
                    columnShifts = 0;

                    result = 'Canceled';
                    return
                else
                    waitbar(0, progressBar, 'Aligning full image set...');
                end
            end
            
            phaseDifferences = zeros(1, Info.maxIndex + 1);
            rowShifts = zeros(1, Info.maxIndex + 1);
            columnShifts = zeros(1, Info.maxIndex + 1);
            
            for f = 0:Info.maxIndex
                frame = sbxRead(Info, f);

                if any(frameCrop > 0)
                    frame = frame(frameCrop(3) + 1:Info.sz(1) - frameCrop(4), frameCrop(1) + 1:Info.sz(2) - frameCrop(2));
                end
            
                if gaussianFilter > 0
                    frame = imgaussfilt(frame, gaussianFilter);
                end

                [temp, ~] = dftregistration(fft2(reference), fft2(frame), subpixelFactor);
                
                phaseDifferences(f + 1) = temp(2);
                rowShifts(f + 1) = temp(3);
                columnShifts(f + 1) = temp(4);
                
                if rowShifts(f + 1) > 15 || columnShifts(f + 1) > 15
                    warning(['Large frame-shift detected at frame ', int2str(f), '. Frame could be misaligned.']);
                end
                
                if GUI
                    if getappdata(progressBar, 'Canceling')
                        delete(progressBar);
                
                        phaseDifferences = 0;
                        rowShifts = 0;
                        columnShifts = 0;
                
                        result = 'Canceled';
                        return
                    else
                        waitbar((f + 1)/(Info.maxIndex + 1), progressBar);
                    end
                end
            end
        end
    end
    
    save([Info.Directory.folder, Info.Directory.name, '.rigid'], 'phaseDifferences', 'rowShifts', 'columnShifts', 'frameCrop');

    delete(progressBar);
    
    result = 'Completed';
    
end