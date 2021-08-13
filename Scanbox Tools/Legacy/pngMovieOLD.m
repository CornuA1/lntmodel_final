function pngMovie(frameRate)
    
    %PNGMOVIE Generate a movie from .png files.
    %   PNGMOVIE() writes a movie from the .png files in the designated directory.
    
    % specify cropping in pixels: [from left, from right, from top, from bottom]
    %frameCrop = [95, 0, 5, 0]; % somas
    %frameCrop = [95, 0, 5, 5]; % dendrites
    frameCrop = [0, 0, 0, 0];
    
    % set number of pixels to correct resonance offset for bidirectional scanning
    resonanceOffset = 0;
    
    % indicate if we want to skip over any frames (defaults are 1 and 1)
    frameOffset = 1;
    frameStep = 1;
    
    % indicate if we want motion-corrected frames
    motionCorrected = 1;
    
    % ask user to identify imaging data directory
    try
        [pngPath] = uigetdir('Please select folder containing imaging data.');
    catch
        waitfor(msgbox('Error: Please select valid folder.'));
        error('Please select valid folder.');
    end
    
    % pull off the file extension
    movieName = [pngPath, '.mp4'];
    
    % create movie
    try
        imagingMovie = VideoWriter(movieName, 'MPEG-4');
    catch
        waitfor(msgbox('Error: Cannot create imaging movie with given filename.'));
        error('Cannot create imaging movie with given filename.');
    end
    
    % set quality (percentage)
    imagingMovie.Quality = 100;
    imagingMovie.FrameRate = frameRate;
    open(imagingMovie);
    
    % reassure user
    disp('Saving imaging movie...');
    
    % iterate through frames using read function
    for f = frameOffset:frameStep:numel(dir([pngPath, '\*.png']))
        
        % reassure user
        if f == uint32(numel(dir([pngPath, '\*.png']))/2)
            disp('50% complete...');
        end

        % call next frame and do some manipulations to make nice
        if motionCorrected == 1
            frame = imread([pngPath, '\registered_', int2str(f), '.png']);
        else
            frame = imread([pngPath, '\', int2str(f), '.png']);
        end
        
        if resonanceOffset ~= 0
            for j = 2:2:size(frame, 1)
                frame(j, 1:end) = [frame(j, 1 + resonanceOffset:end), zeros(1, resonanceOffset)];
            end
        end
        
        frame = frame(frameCrop(3) + 1:size(frame, 1) - frameCrop(4), frameCrop(1) + 1:size(frame, 2) - frameCrop(2));
        
        % write to movie
        writeVideo(imagingMovie, double(frame)/65535);
    end
    
    close(imagingMovie);
    
    % reassure user
    disp('Done.');
   
end