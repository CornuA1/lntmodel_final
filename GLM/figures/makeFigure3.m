function makeFigure3(coef)
figure;
for x = 1:10
    subplot(10,1,x);
    plot(coef(:,x));
end
figure;
for x = 11:20
    subplot(10,1,(x-10));
    plot(coef(:,x));
end
figure;
for x = 21:28
    subplot(8,1,(x-20));
    plot(coef(:,x));
end
end