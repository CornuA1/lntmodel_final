function frames = sbxgrabframe(fname,ind,num)
% Retrieve frames from a raw .sbx datafile generated by NLW
%   fname   - filename (don't include suffix)
%   ind     - index from which to read
%   num     - number of frames to grab. -1 means until end of file.
%
%   Most of this function is derived from sbxread provided by Neurolabware
%
%   Author: Lukas Fischer
%

mi = load(fname);   % load file information    
mi.info.fid = fopen([fname '.sbx']); % open corresponding .sbxfile
d = dir([fname '.sbx']);

switch mi.info.channels % set number of channels
    case 1
        mi.info.nchan = 2;      % both PMT0 & 1
        factor = 1;
    case 2
        mi.info.nchan = 1;      % PMT 0
        factor = 2;
    case 3
        mi.info.nchan = 1;      % PMT 1
        factor = 2;
end

if(mi.info.scanmode==0)
	recordsPerBuffer = mi.info.recordsPerBuffer*2;
else
    recordsPerBuffer = mi.info.recordsPerBuffer;
end
if isfield(mi.info,'scanbox_version') && mi.info.scanbox_version >= 2
    mi.info.max_idx =  d.bytes/recordsPerBuffer/mi.info.sz(2)*factor/4 - 1;
    mi.info.nsamples = (mi.info.sz(2) * recordsPerBuffer * 2 * mi.info.nchan);   % bytes per record 
else
    mi.info.max_idx =  d.bytes/mi.info.bytesPerBuffer*factor - 1;
end

if(isfield(mi.info,'fid') && mi.info.fid ~= -1)
    if num==-1
        num=mi.info.max_idx-ind+1;
    end
    fseek(mi.info.fid,ind*mi.info.nsamples,'bof');
    frames = fread(mi.info.fid,mi.info.nsamples/2 * num,'uint16=>uint16');
    frames = reshape(frames,[mi.info.nchan mi.info.sz(2) recordsPerBuffer  num]);
    frames = intmax('uint16')-permute(frames,[1 3 2 4]);
    fclose(mi.info.fid);
else
    frames = [];
    fclose(mi.info.fid);
end

end