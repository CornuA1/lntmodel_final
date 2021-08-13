function makeFigure2(coef)
figure;
pos2 = [0.1 0.4 0.8 0.25];
subplot('Position',pos2)
scatter([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28], coef, 'r')
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28])
xticklabels({'-180','-160','-140','-120','-100','-80','-60','-40',...
    '-20','0','20','40','60','80','100','120','140','160',...
    'Slow Speed','Fast Speed','Linear Speed','Lick Location','Reward Event',...
    'Reward Event -30','Reward Event +30','Trial Onset','Trial Onset +30','Trial Onset +60'})
xtickangle(45)
%pos2 = [0.1 0.1 0.8 0.25];
%subplot('Position',pos2)
%plot(linspace(1, size(test_data,1),size(test_data,1)), test_data, linspace(1, size(pred_data,1),size(pred_data,1)), pred_data*15, 'r')
end