function [predictor_short_x, location_vector] = makeSpatialPred()
TRACK_START = 0;
BINNR_SHORT = 80;
BINNR_LONG = 100;
TRACK_END_SHORT = 400;
TRACK_END_LONG = 500;
TRACK_SHORT = 3;
TRACK_LONG = 4;
NUM_PREDICTORS = 20;

binsize_short = TRACK_END_SHORT/NUM_PREDICTORS;
% matrix that holds predictors
predictor_short_x = zeros(TRACK_END_SHORT, NUM_PREDICTORS);
% calculate bin edges and centers
boxcar_short_centers = linspace(TRACK_START+binsize_short/2, TRACK_END_SHORT-binsize_short/2, NUM_PREDICTORS);
location_vector = 0:1:TRACK_END_SHORT-1;

% gaussian kernel to convolve boxcar spatial predictors with
n = 100;
stdev = 5;
alpha = (n-1)/(2*stdev);
gauss_kernel = gausswin(n,alpha) * 2;

%create boxcar vectors and convolve with gaussian
    for i = 1:length(boxcar_short_centers)
        % calculate single predictor
        predictor_x = zeros(1,TRACK_END_SHORT);
        start = int16(boxcar_short_centers(i)-binsize_short/2 + 1);
        fin = int16(boxcar_short_centers(i)+binsize_short/2);
        predictor_x(start : fin) = 1;
        predictor_x = conv(predictor_x,gauss_kernel,'same');
        predictor_x = predictor_x/max(predictor_x);
        predictor_short_x(:,i) = predictor_x;
        % predictor_short_x[:,i] = signal.resample(predictor_x,BINNR_SHORT)
    end
%add constant predictor
%predictor_short_x = np.insert(predictor_short_x,0,np.ones((predictor_short_x.shape[0])), axis=1)
combineSpatPred = false;
if combineSpatPred
    combineSpatialPred();
end
end