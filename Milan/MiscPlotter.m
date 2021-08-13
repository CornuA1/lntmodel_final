closed_loop_file = 'ClosedLoopResults.mat';
open_loop_file = 'OpenLoopResults.mat';
addpath('misc_scripts')
%%
subplot(2,2,1)
load(closed_loop_file)
ppos_lng(isnan(rpos_lng))=[];
rpos_lng(isnan(rpos_lng))=[];
corr_segs = [150 200;200 250;250 300;300 350];%segments over which to computer correlations
r_eb = zeros(3,size(corr_segs,1));
for k = 1 : size(corr_segs,1)
    inds = intersect(find(rpos_lng<=corr_segs(k,2)),find(rpos_lng>=corr_segs(k,1)));
    n(k) = length(inds);
    [R,P,RL,RU] = corrcoef(rpos_lng(inds),ppos_lng(inds));
    r_eb(1,k) = R(1,2); r_eb(2,k) = RU(1,2); r_eb(3,k) = RL(1,2);
end
bar(1:size(corr_segs,1),r_eb(1,:)), hold on, errorbar(1:size(corr_segs,1),r_eb(1,:),r_eb(2,:)-r_eb(1,:),'k.')
ylim([-.05 .45])
title('long trials')
ylabel('closed loop','fontweight','bold')
r_diff = nan(length(corr_segs));
for k = 1:length(corr_segs)
    for kk = 1:length(corr_segs)
        n1=n(k); n2=n(kk);
        r1=r_eb(1,k); r2=r_eb(1,kk);
        r_diff(k,kk) = ((0.5*log((1+r1)/(1-r1)))-(0.5*log((1+r2)/(1-r2))))/sqrt((1/(n1-3))+(1/(n2-3)));
    end
end
r_diff = normcdf(-abs(r_diff))*2;
crit = .05 / ( (length(corr_segs)*(length(corr_segs)-1))/2 );%bonferonni correct

ppos_shrt(isnan(rpos_shrt))=[];
rpos_shrt(isnan(rpos_shrt))=[];
r_eb = zeros(3,size(corr_segs,1));
for k = 1 : size(corr_segs,1)
    inds = intersect(find(rpos_shrt<=corr_segs(k,2)),find(rpos_shrt>=corr_segs(k,1)));
    n(k) = length(inds);
    [R,P,RL,RU] = corrcoef(rpos_shrt(inds),ppos_shrt(inds));
    r_eb(1,k) = R(1,2); r_eb(2,k) = RU(1,2); r_eb(3,k) = RL(1,2);
end

subplot(2,2,2)
bar(1:size(corr_segs,1),r_eb(1,:)), hold on, errorbar(1:size(corr_segs,1),r_eb(1,:),r_eb(2,:)-r_eb(1,:),'k.')
ylim([-.05 .45])
title('short trials')


subplot(2,2,3)
load(open_loop_file)
ppos_lng(isnan(rpos_lng))=[];
rpos_lng(isnan(rpos_lng))=[];
r_eb = zeros(3,size(corr_segs,1));
for k = 1 : size(corr_segs,1)
    inds = intersect(find(rpos_lng<=corr_segs(k,2)),find(rpos_lng>=corr_segs(k,1)));
    n(k) = length(inds);
    [R,P,RL,RU] = corrcoef(rpos_lng(inds),ppos_lng(inds));
    r_eb(1,k) = R(1,2); r_eb(2,k) = RU(1,2); r_eb(3,k) = RL(1,2);
end
bar(1:size(corr_segs,1),r_eb(1,:)), hold on, errorbar(1:size(corr_segs,1),r_eb(1,:),r_eb(2,:)-r_eb(1,:),'k.')
ylim([-.05 .2])
ylabel('open loop','fontweight','bold')

ppos_shrt(isnan(rpos_shrt))=[];
rpos_shrt(isnan(rpos_shrt))=[];
r_eb = zeros(3,size(corr_segs,1));
for k = 1 : size(corr_segs,1)
    inds = intersect(find(rpos_shrt<=corr_segs(k,2)),find(rpos_shrt>=corr_segs(k,1)));
    n(k) = length(inds);
    [R,P,RL,RU] = corrcoef(rpos_shrt(inds),ppos_shrt(inds));
    r_eb(1,k) = R(1,2); r_eb(2,k) = RU(1,2); r_eb(3,k) = RL(1,2);
end
subplot(2,2,4)
bar(1:size(corr_segs,1),r_eb(1,:)), hold on, errorbar(1:size(corr_segs,1),r_eb(1,:),r_eb(2,:)-r_eb(1,:),'k.')
ylim([-.05 .2])

%%
load(closed_loop_file)
figure
subplot(2,2,1), scatter(ppos_shrt,rpos_shrt,'k.'), xlim([150 350]), ylim([150 350]), axis square
title('closed loop')
ylabel('short trials','fontweight','bold')
load(open_loop_file)
subplot(2,2,2), scatter(ppos_shrt,rpos_shrt,'k.'), xlim([150 350]), ylim([150 350]), axis square
title('open loop')
corr(ppos_shrt,rpos_shrt)
load(closed_loop_file)
subplot(2,2,3), scatter(ppos_lng,rpos_lng,'k.'), xlim([150 400]), ylim([150 400]), axis square
ylabel('long trials','fontweight','bold')
load(open_loop_file)
subplot(2,2,4), scatter(ppos_lng,rpos_lng,'k.'), xlim([150 400]), ylim([150 400]), axis square
corr(ppos_lng,rpos_lng)
%%
figure
load(closed_loop_file)
subplot(2,2,1), binned_plot(rpos_lng,ppos_lng,25), xlim([150 400]), ylim([150 400]), hold on, plot(50:.5:400,50:.5:400)
title('closed loop, long')
subplot(2,2,3), binned_plot(rpos_shrt,ppos_shrt,25), xlim([150 350]), ylim([150 350]), hold on, plot(50:.5:350,50:.5:350)
title('closed loop, short')
load(open_loop_file)
subplot(2,2,2), binned_plot(rpos_lng,ppos_lng,25), xlim([150 400]), ylim([150 400]), hold on, plot(50:.5:400,50:.5:400)
title('open loop, long')
subplot(2,2,4), binned_plot(rpos_shrt,ppos_shrt,25), xlim([150 350]), ylim([150 350]), hold on, plot(50:.5:350,50:.5:350)
title('open loop, short')
%%
load(closed_loop_file)
figure
[ml,xll,xml]=scatterbin(ppos_lng,rpos_lng,50,150,400);
subplot(2,2,1)
imagesc(xll,xll,ml,[0 200]), axis square
title('closed loop, long')
subplot(2,2,3)
plot(xll,xml), xlim([150 400])
load(open_loop_file)
[ms,xll,xms]=scatterbin(ppos_lng,rpos_lng,50,150,350);
subplot(2,2,2)
imagesc(xll,xll,ms,[0 200]), axis square
title('open loop, long')
subplot(2,2,4)
plot(xll,xms), xlim([150 350])
%%
load(closed_loop_file)
figure
[ml,xll,xml]=scatterbin(ppos_shrt,rpos_shrt,50,150,350);
subplot(2,2,1)
imagesc(xll,xll,ml,[0 200]), axis square
title('closed loop, short')
subplot(2,2,3)
plot(xll,xml), xlim([150 350])
load(open_loop_file)
[ms,xll,xms]=scatterbin(ppos_shrt,rpos_shrt,50,150,350);
subplot(2,2,2)
imagesc(xll,xll,ms,[0 200]), axis square
title('open loop, short')
subplot(2,2,4)
plot(xll,xms), xlim([150 350])