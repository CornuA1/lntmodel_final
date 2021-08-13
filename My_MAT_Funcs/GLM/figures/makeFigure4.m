function makeFigure4(behave)
figure;
for x = 1:size(behave,2)
    plot(behave(:,x));
    hold on
end
hold off
end