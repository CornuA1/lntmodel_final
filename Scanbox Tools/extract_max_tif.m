function extract_max_tif()
[sbxName, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.');
sbxName = strtok(sbxName, '.');
load([sbxPath, sbxName, '.pre'], '-mat', 'meanImage', 'maxIntensityImage', 'crossCorrelationImage', 'pcaImage')
maxIntensityImage = maxIntensityImage/double(intmax('uint16'));
disp('exporting tif...');    
imwrite(maxIntensityImage,[sbxPath sbxName '_max.tif'],'tif','writemode','append');
disp(['exported ' sbxPath sbxName  '.tif']);
end