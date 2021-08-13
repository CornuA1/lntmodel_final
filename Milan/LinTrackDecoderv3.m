clear
addpath('Data'), addpath('misc_scripts')
fs={'matlab_LF161202_1_Day20170209_l5_long_all.mat'
    'matlab_LF161202_1_Day20170209_l5_short_all.mat'
    'matlab_LF161202_1_Day20170209_l23_long_all.mat'
    'matlab_LF161202_1_Day20170209_l23_short_all.mat'
    'matlab_LF170110_2_Day201748_1_long_all.mat'
    'matlab_LF170110_2_Day201748_1_short_all.mat'
    'matlab_LF170110_2_Day201748_2_long_all.mat'
    'matlab_LF170110_2_Day201748_2_short_all.mat'
    'matlab_LF170110_2_Day201748_3_long_all.mat'
    'matlab_LF170110_2_Day201748_3_short_all.mat'
%     'matlab_LF170223_1_Day201776_long_all.mat'
%     'matlab_LF170223_1_Day201776_short_all.mat'
    'matlab_LF170420_1_Day201783_long_all.mat'
    'matlab_LF170420_1_Day201783_short_all.mat'
    'matlab_LF170420_1_Day2017719_long_all.mat'
    'matlab_LF170420_1_Day2017719_short_all.mat'
    'matlab_LF170421_2_Day2017720_long_all.mat'
    'matlab_LF170421_2_Day2017720_short_all.mat'
    'matlab_LF170421_2_Day20170719_long_all.mat'
    'matlab_LF170421_2_Day20170719_short_all.mat'
    'matlab_LF170613_1_Day20170804_long_all.mat'
    'matlab_LF170613_1_Day20170804_short_all.mat'};
%     'matlab_LF171212_2_Day2018218_2_long_all.mat'
%     'matlab_LF171212_2_Day2018218_2_short_all.mat'};

ppos_long = [];
rpos_long = [];
ppos_short=[]; rpos_short=[];

exclude_stff=1;

for fi = 1:length(fs)%[1:5 7:11]%1:length(fs)
    %load data
    load(fs{fi})
    % create matrix 'trial', which stores start and end indices
    % for each trial / black box period, along with a lable
    % which denotes whether the trial is short, long or black box
    tr = find(diff(behavior(:,4)))'; tr = [1 tr length(behavior)];
    trl = zeros(length(tr)./2,3);
    ind = 1;
    for k=1:2:length(tr)
        trl(ind,1) = tr(k)+1; trl(ind,2) = tr(k+1); trl(ind,3) = behavior(tr(k)+1,4);
        ind = ind+1;
    end
    
    if exclude_stff
    
    % find when less than 150
    
    sub_150 = find(behavior(:,2)<150);
    
    % find when immobile        
    c = contiguous( behavior(:,3) < 1 , 1);%find contiguous runs 
        
    c = c{2};
        
    immob_segs = find( ( c(:,2)-c(:,1) ) > 60 );
        
    immob_inds=[];
        
    for kk = immob_segs
        immob_inds = [immob_inds c(kk,1):c(kk,2)];
    end
        
    % combine the two, set to NaN
    
    rej_inds = union(immob_inds,sub_150);
    
    behavior(rej_inds,2) = NaN;
    
    end
    
    %% fit data
    x0ta_lng=[]; xva_lng=[];
    x0ta_shrt=[]; xva_shrt=[];
    % iterate thru each trial we're leaving out
    for k = 1:length(trl)
        trn=[]; trl_type = trl(k,3);
        for kk = 1:length(trl)
            if k~=kk && trl_type==trl(kk,3)
                trn = [trn trl(kk,1):trl(kk,2)];
            end
        end
        
        %get the position data for the validation trial
        xvi = trl(k,1):trl(k,2);
        xv = behavior(xvi,2);
        
        %fit the model
        Mdl = fitrtree(df(trn,:),behavior(trn,2));
        %predict the left out trial
        x0t = predict(Mdl,df(xvi,:));
        
        %save
        if trl_type==3
            x0ta_shrt = [x0ta_shrt; x0t];
            xva_shrt = [xva_shrt; xv];
        else
            x0ta_lng = [x0ta_lng; x0t];
            xva_lng = [xva_lng; xv];
        end

    end
    
    %save in separate vector depending on if it's a 
    %short or long trial file
    if rem(fi,2)==0
    ppos_short = [ppos_short; x0ta_shrt];
    rpos_short = [rpos_short; xva_shrt];
    else
    ppos_long = [ppos_long; x0ta_lng];
    rpos_long = [rpos_long; xva_lng];
    end
    
    Res.x0ta_lng{fi}=x0ta_lng;
    Res.xva_lng{fi}=xva_lng;

    Res.x0ta_shrt{fi}=x0ta_shrt;
    Res.xva_shrt{fi}=xva_shrt;
    disp(fi)
end

save('ClosedLoopResults.mat','Res','ppos_short','rpos_short','rpos_long','ppos_long')