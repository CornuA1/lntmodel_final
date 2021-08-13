clear;
load('Q:\Documents\Harnett UROP\LF191022_1\20191209\M01_000_000_results.mat');
y = new_dF(1,:)';
X = ones(length(y),1);
b = robustfit(X,y);