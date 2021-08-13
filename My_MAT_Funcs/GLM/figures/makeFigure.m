function makeFigure(fit_data, fit_pred, coef, test_data, pred_data)
figure;
%plot
subplot(5,1,1)
plot(fit_data)
subplot(5,1,2)
plot(fit_pred*10, 'r')
subplot(5,1,3)
scatter([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31], coef, 'g')
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31])
xticklabels({'Intercept','Bin 1','Bin 2','Bin 3','Bin 4','Bin 5','Bin 6','Bin 7','Bin 8','Bin 9','Bin 10',...
    'Bin 11','Bin 12','Bin 13','Bin 14','Bin 15','Bin 16','Bin 17','Bin 18','Bin 19','Bin 20',...
    'Const Speed Pred','Slow Speed','Fast Speed','Percentage Speed','In Landmark','Begin Landmark',...
    'End Landmark','Reward Event','Reward Event For Offset','Reward Event Back Offset'})
xtickangle(45)
subplot(5,1,4)
plot(test_data)
subplot(5,1,5)
plot(pred_data*10, 'r')
end
%function makeFigure(predictor_short_x, coef, roi_gcamp, test, pred)
%plot
%subplot(5,1,1)
%plot(predictor_short_x)
%subplot(5,1,2)
%plot(coef)
%subplot(5,1,3)
%plot(roi_gcamp, 'g')
%subplot(5,1,4)
%hold on
%plot(test,'g')
%plot(pred,'r')
%hold off
%subplot(5,1,5)
%plot(test-pred)
%end