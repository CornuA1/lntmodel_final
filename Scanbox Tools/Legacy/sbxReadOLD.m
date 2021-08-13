function frame = sbxRead(Info, index)

    %SBXREAD Read individual frames from .sbx file.
    %   frame = SBXREAD(Info, index) gives the frame in the indicated index.
    %
    %   Info: structure
    %       Structure containing important details about imaging data.
    %
    %   index: integer
    %       Index of the first frame to be read. The first index is 0.
    %
    %   frame: uint16
    %       Frame with dimensions [Info.sz(1), Info.sz(2)].
    
    fileID = fopen([Info.Directory.folder, '\', Info.Directory.name], 'r');
    
    % set access point to start of indicated frame in bytes
    fseek(fileID, index*Info.bytesPerFrame, 'bof');
    
    % read frame in terms of samples
    frame = fread(fileID, Info.samplesPerFrame, 'uint16=>uint16');
    
    if ~isempty(frame)
        frame = reshape(frame, [Info.nChannels, Info.sz(2), Info.sz(1)]);
    else
        waitfor(msgbox('Error: Empty frame. Index range likely outside of bounds.'));
        error('Empty frame. Index range likely outside of bounds.');
    end
        
    frame = squeeze(intmax('uint16') - permute(frame, [1, 3, 2]));

    fclose(fileID);
    
end