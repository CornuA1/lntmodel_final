clear;

% createGLMdata('D:\Lukas\data\animals_raw\LF191024_1\20191114\', 'M01_000_000_results.mat', 'calcium','gaussian');
createGLMdata('D:\Lukas\data\animals_raw\LF191022_1\20191204\', 'M01_000_002_results.mat', 'calcium','gaussian');
create_glm_regressors('D:\Lukas\data\animals_raw\LF191022_1\20191204\', 'M01_000_002_results.mat')