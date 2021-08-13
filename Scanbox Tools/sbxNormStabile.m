function sbxNormStabile(sbxPath, sbxName)
%
% Brief description:
%       Aligns the 2-photon imaging file using the NoRMCorre package
%
% Inputs:
%       sbxPath - Location of the 2-photon .sbx and .mat file
%                   'D:\Lukas\20191211\M01\'
%       sbxName - Name of the 2-photon .sbx and .mat file
%                   'M01_000_001'
% Outputs:
%       '[sbxName]_nr.h5' - file of the aligned 2-photon data
%

%[sbxName, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.');
addpath(sbxPath);
%sbxName = strtok(sbxName, '.');
info = sbxInfo([sbxPath, sbxName]);
disp('File found.');
%%
output_type = 'h5';
append = '_nr';
end_f = info.maxIndex;
Parameters.frameCrop(1) = 0;
Parameters.frameCrop(2) = 0;
Parameters.frameCrop(3) = 0;
Parameters.frameCrop(4) = 0;
Y_n = sbxreadframes_info(sbxName,info,0,end_f,Parameters);
Y = squeeze(Y_n);
clear Y_n;
%Y = single(Y);
%Y = Y - min(Y(:));
disp('File indexed and compressed.');
%%
output_filename = fullfile(sbxPath,[sbxName,append,'.',output_type]);
options_mc = NoRMCorreSetParms('d1',size(Y,1),'d2',size(Y,2),'grid_size',[128,128],'init_batch',500,...
                'overlap_pre',32,'mot_uf',4,'bin_width',500,'max_shift',24,'max_dev',8,'us_fac',50,...
                'output_type',output_type,'correct_bidir',false,'output_filename',output_filename);
[M2,shifts2,template2,options_nonrigid] = normcorre_batch(Y,options_mc);
disp('File aligned.');
disp(sbxName);
end