%%

baselinePercentile = 5;
windowRadius = 60;
fs = 15.5;

% number of samples in window
win_samples = windowRadius * fs;

try
    [sbxName, sbxPath] = uigetfile('.sig', 'Please select file signal file');
catch
    waitfor(msgbox('Error: Please select valid .sig file.'));
    error('Please select valid .sig file.');
end

%%

temp = csvread([sbxPath, sbxName]);


neuropilValues = temp(:, size(temp, 2)/3 + 1:2*size(temp, 2)/3)';
neuropilValues = neuropilValues(1,:);

roiValues = temp(:, 2*size(temp, 2)/3 + 1:end)';
roiValues = roiValues(1,:);

numberOfROIs = size(roiValues, 1);

% first, calculate F0, estimated as the (baselinePercentile)th percentile fluorescence value within a (2*windowRadius) second window
roiBaselines = zeros(size(roiValues));
neuropilBaselines = zeros(size(roiValues));

coefficients = robustfit(neuropilValues, roiValues,'huber');
neuropilCorrection = coefficients(2);

figure;
scatter(neuropilValues,roiValues);
hold on;
plot(neuropilValues,coefficients(1)+coefficients(2)*neuropilValues,'r','LineWidth',2);
hold off;

% for b = 1:size(roiValues,2)
%     if frameTimes(b) - frameTimes(1) < windowRadius
%         finish = find(frameTimes <= frameTimes(1) + 2*windowRadius);
%         finish = finish(end);
% 
%         % don't include the first frame in this calculation - shit's all sorts of messed up
%         roiBaselines(:, b) = prctile(roiValues(:, 2:finish), baselinePercentile, 2);
%         neuropilBaselines(:, b) = prctile(neuropilValues(:, 2:finish), baselinePercentile, 2);
%     elseif frameTimes(end) - frameTimes(b) < windowRadius
%         roiBaselines(:, b) = prctile(roiValues(:, frameTimes >= frameTimes(end) - 2*windowRadius), baselinePercentile, 2);
%         neuropilBaselines(:, b) = prctile(neuropilValues(:, frameTimes >= frameTimes(end) - 2*windowRadius), baselinePercentile, 2);
%     else
%         start = find(frameTimes >= frameTimes(b) - windowRadius);
%         start = start(1);
%         finish = find(frameTimes <= frameTimes(b) + windowRadius);
%         finish = finish(end);
% 
%         roiBaselines(:, b) = prctile(roiValues(:, start:finish), baselinePercentile, 2);
%         neuropilBaselines(:, b) = prctile(neuropilValues(:, start:finish), baselinePercentile, 2);
%     end
% end
% 
% dFs = zeros(size(roiValues));
% neuropildFs = zeros(size(neuropilValues));
% 
% for ROI = 1:numberOfROIs
%     dFs(ROI, :) = (roiValues(ROI, :) - roiBaselines(ROI, :))./roiBaselines(ROI, :);
%     neuropildFs(ROI, :) = (neuropilValues(ROI, :) - neuropilBaselines(ROI, :))./neuropilBaselines(ROI, :);
% 
%     coefficients = robustfit(neuropildFs(ROI, :), dFs(ROI, :));
%     
%     neuropilCorrection = coefficients(2);
%     
%     dFs(ROI, :) = dFs(ROI, :) - neuropilCorrection*neuropildFs(ROI, :);
% end

disp('done');
