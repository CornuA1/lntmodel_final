clear;

load('Q:\Documents\Harnett UROP\LF191022_1\20191209\glm_files.mat');

figure;
subplot(4,1,1);
plot(predictor_long_x_fit(1000:2000,4)); hold on; %-120
plot(predictor_long_x_fit(1000:2000,8)); hold on; %-40
plot(predictor_long_x_fit(1000:2000,12)); hold on; %+40
plot(predictor_long_x_fit(1000:2000,16)); hold off;%+120
subplot(4,1,2);
plot(predictor_long_x_fit(1000:2000,19)); hold on;
plot(predictor_long_x_fit(1000:2000,21)); hold off;
subplot(4,1,3);
plot(predictor_long_x_fit(1000:2000,22));
subplot(4,1,4);
plot(predictor_long_x_fit(1000:2000,23)); hold on;
plot(predictor_long_x_fit(1000:2000,28)); hold off;