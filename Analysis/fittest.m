x = (1:10)';
rng default; % For reproducibility
y = 10 + 2*x + randn(10,1);
y(10) = 0;

brob = robustfit(x,y);

scatter(x,y,'filled'); grid on; hold on
plot(x,brob(1)+brob(2)*x,'g','LineWidth',2)
legend('Data','Robust Regression')
hold off