function x = sbxreadframes_info(fname,info,k,N,Parameters)

% img = sbxread(fname,k,N,varargin)
%
% Reads from frame k to k+N-1 in file fname
% 
% fname - the file name (e.g., 'xx0_000_001')
% k     - the index of the first frame to be read.  The first index is 0.
% N     - the number of consecutive frames to read starting with k.
%
% If N>1 it returns a 4D array of size = [#pmt rows cols N] 
% If N=1 it returns a 3D array of size = [#pmt rows cols]
%
% #pmts is the number of pmt channels being sampled (1 or 2)
% rows is the number of lines in the image
% cols is the number of pixels in each line
%
% The function also creates a global 'info' variable with additional
% informationi about the file

info.nchan = 1;      % PMT 0
info.fid = fopen([fname '.sbx']);
d = dir([fname '.sbx']);
info.nsamples = (info.sz(2) * info.recordsPerBuffer * 2 * info.nchan);   % bytes per record 

if isfield(info,'scanbox_version') && info.scanbox_version >= 2
    info.nsamples = (info.sz(2) * info.recordsPerBuffer * 2 * info.nchan);   % bytes per record 
end

if(isfield(info,'fid') && info.fid ~= -1)
    
    % nsamples = info.postTriggerSamples * info.recordsPerBuffer;
        
    try
        fseek(info.fid,k*info.nsamples,'bof');
        x = fread(info.fid,info.nsamples/2 * N,'uint16=>uint16');
        x = reshape(x,[info.nchan info.sz(2) info.recordsPerBuffer  N]);
    catch
        error('Cannot read frame.  Index range likely outside of bounds.');
    end

    x = intmax('uint16')-permute(x,[1 3 2 4]);
    
    if any(Parameters.frameCrop > 0) % crop frames if so desired
        frameCrop = Parameters.frameCrop;
        num_frames = size(x,4);
        y = zeros(size(x,1), info.sz(1)-frameCrop(3)-frameCrop(4), info.sz(2)-frameCrop(1)-frameCrop(2),size(x,4));
        for i=1:num_frames
            y(1,:,:,i) = x(1,frameCrop(3) + 1:info.sz(1) - frameCrop(4), frameCrop(1) + 1:info.sz(2) - frameCrop(2),i);
        end
        x = y;
    end
else
    x = [];
end