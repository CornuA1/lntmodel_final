function sbxMovie(sbxName)
    
    %SBXMOVIE Generate a movie from .sbx file.
    %   SBXMOVIE(sbxName) writes a movie from the frames in the selected .sbx file.
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
    
    rollingWindowSize = 2;
    
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
    
    %frameTimes = Info.timeStamps(frameOffset + 1:frameStep:Info.maxIndex);
    frameTimes = (frameIndices*512)/(Info.resfreq*(2 - Info.scanmode));

    frameRate = length(frameTimes)/(frameTimes(end) - frameTimes(1));
    
    movieName = [sbxPath, sbxName, '.mp4'];
    
    try
        imagingMovie = VideoWriter(movieName, 'MPEG-4');
    catch
        waitfor(msgbox('Error: Cannot create imaging movie with given filename.'));
        error('Cannot create imaging movie with given filename.');
    end
    
    % movie quality is in percentage
    imagingMovie.Quality = 100;
    imagingMovie.FrameRate = frameRate;
    open(imagingMovie);
    
    disp('Saving imaging movie...');
    
    for f = 1:length(frameIndices)
        if f == uint32(length(frameIndices)/2)
            disp('50% complete...');
        end

        frames = zeros(Info.sz(1), Info.sz(2), rollingWindowSize*2 + 1);

        if frameIndices(f) - rollingWindowSize < 0
            start = 0;
            finish = rollingWindowSize*2;
        elseif frameIndices(f) + rollingWindowSize > Info.maxIndex
            start = Info.maxIndex - rollingWindowSize*2;
            finish = Info.maxIndex;
        else
            start = frameIndices(f) - rollingWindowSize;
            finish = frameIndices(f) + rollingWindowSize;
        end

        for i = 0:finish - start
            frames(:, :, i  + 1) = sbxRead(Info, start + i);
        end

        frame = mean(frames, 3);
        
        %frame = sbxRead(Info, frameIndices(f));
        
        % apply resonance offset to every other line
        if resonanceOffset ~= 0
            for j = 2:2:size(frame, 1)
                frame(j, 1:end) = [frame(j, 1 + resonanceOffset:end), zeros(1, resonanceOffset)];
            end
        end
        
        % crop as indicated
        frame = frame(frameCrop(3) + 1:Info.sz(1) - frameCrop(4), frameCrop(1) + 1:Info.sz(2) - frameCrop(2));
        
        writeVideo(imagingMovie, double(frame)/double(intmax('uint16')));
    end
    
    close(imagingMovie);
    
    disp('Done.');
   
end