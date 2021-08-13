clear
fs={'matlab_LF170110_2_Day201748_openloop_1_short_active_all_openloop.mat'
    'matlab_LF170110_2_Day201748_openloop_1_long_active_all_openloop.mat'
    'matlab_LF170110_2_Day201748_openloop_2_short_active_all_openloop.mat'
    'matlab_LF170110_2_Day201748_openloop_2_long_active_all_openloop.mat'
    'matlab_LF170110_2_Day201748_openloop_3_short_active_all_openloop.mat'
    'matlab_LF170110_2_Day201748_openloop_3_long_active_all_openloop.mat'
    'matlab_LF170222_1_Day201776_openloop_short_active_all_openloop.mat'
    'matlab_LF170222_1_Day201776_openloop_long_active_all_openloop.mat'
    'matlab_LF170420_1_Day201783_openloop_short_active_all_openloop.mat'
    'matlab_LF170420_1_Day201783_openloop_long_active_all_openloop.mat'
    'matlab_LF170420_1_Day2017719_openloop_short_active_all_openloop.mat'
    'matlab_LF170420_1_Day2017719_openloop_long_active_all_openloop.mat'
%     'matlab_LF170421_2_Day2017720_openloop_short_active_all_openloop.mat'
%     'matlab_LF170421_2_Day2017720_openloop_long_active_all_openloop.mat'
    'matlab_LF170421_2_Day20170719_openloop_short_active_all_openloop.mat'
    'matlab_LF170421_2_Day20170719_openloop_long_active_all_openloop.mat'
    'matlab_LF170613_1_Day20170804_openloop_short_active_all_openloop.mat'
    'matlab_LF170613_1_Day20170804_openloop_long_active_all_openloop.mat'
    'matlab_LF171212_2_Day2018218_openloop_2_short_active_all_openloop.mat'
    'matlab_LF171212_2_Day2018218_openloop_2_long_active_all_openloop.mat'
    };
ppos_lng=[]; rpos_lng=[];
ppos_shrt=[]; rpos_shrt=[];
exclude_data = 1;
for fi = [1:length(fs)]
    load(fs{fi})
%% remove black box
tr = find(diff(behavior(:,4)))'; tr = [1 tr length(behavior)];
trl = zeros(length(tr)./2,3);
ind = 1;
for k=1:2:length(tr)
    trl(ind,1) = tr(k)+1; trl(ind,2) = tr(k+1); trl(ind,3) = behavior(tr(k)+1,4);
    ind = ind+1;
end
    if exclude_data
    % find when less than 150
    
    sub_150 = find(behavior(:,2)<150);
    
    % find when immobile
    if behavior(:,3) < 1
    c = contiguous( behavior(:,3) < 1 , 1);%find contiguous runs 
        
    c = c{2};
        
    immob_segs = find( ( c(:,2)-c(:,1) ) > 60 );
        
    immob_inds=[];
        
    for kk = immob_segs
        immob_inds = [immob_inds c(kk,1):c(kk,2)];
    end
    else
        immob_inds = [];
    end
        rej_inds = union(immob_inds,sub_150);
    
    behavior(rej_inds,2) = NaN;
    end
%% fit data
x0ta_lng=[]; xva_lng=[];
x0ta_shrt=[]; xva_shrt=[];



mts=[]; mtl=[]; 
for k = 1:length(trl)
    trn=[]; trl_type = trl(k,3);
    for kk = 1:length(trl)
        if k~=kk && trl_type==trl(kk,3)
            trn = [trn trl(kk,1):trl(kk,2)];
        end
    end
    xv = behavior(trl(k,1):trl(k,2),2);
    Mdl = fitrtree(df(trn,:),behavior(trn,2));
    x0t = predict(Mdl,df(trl(k,1):trl(k,2),:));
    if trl_type==3
        x0ta_shrt = [x0ta_shrt; x0t]; 
        xva_shrt = [xva_shrt; xv];
    else
        x0ta_lng = [x0ta_lng; x0t];
        xva_lng = [xva_lng; xv];
    end
    
end

if rem(fi,2)==1
ppos_shrt = [ppos_shrt; x0ta_shrt];
rpos_shrt = [rpos_shrt; xva_shrt];
else
ppos_lng = [ppos_lng; x0ta_lng];
rpos_lng = [rpos_lng; xva_lng];
end

Res.x0ta_lng{fi}=x0ta_lng;
Res.xva_lng{fi}=xva_lng;

Res.x0ta_shrt{fi}=x0ta_shrt;
Res.xva_shrt{fi}=xva_shrt;
disp(fi)
end
save('OpenLoopResults.mat','Res','ppos_shrt','rpos_shrt','rpos_lng','ppos_lng')