%combine planes
clear all
close all
clc

index=[]
for kk=1:150
    index=[index;'000'];
end

for kk=2
    index(kk,3)=num2str(5*(kk-1))
end

for kk=3:20
    index(kk,2:3)=num2str(5*(kk-1))
end

for kk=21:150
    index(kk,1:3)=num2str(5*(kk-1))
end

%%
previousframe=[];
frame=[];
figure;
fullstack=[];

for kk=1:140
    
    indexyo=index(kk,:)


sbxName=[indexyo,'compiledstack']
sbxPath='F:\Quique\pizzaman stack\pizzaman4\'

% pull off the file extension
% sbxName = strtok(sbxName, '.');

Info = sbxInfo([sbxPath, sbxName]);

 load([sbxPath, sbxName, '.pre'], '-mat', 'meanImage', 'maxIntensityImage', 'primage50', 'primage75', 'primage90')
 
%  frame=maxIntensityImage;
%  frame=(primage50);
 
 frame=(meanImage);

 
 if kk>1
 [yoyo, ~] = dftregistration(fft2(previousframe), fft2(frame), 1);
 phaseDifference=yoyo(2);
 rowShift=yoyo(3);
 columnShift=yoyo(4);
 adjustedImage = fft2(frame);
 
 [numberOfRows, numberOfColumns] = size(adjustedImage);
 Nr = ifftshift(-fix(numberOfRows/2):ceil(numberOfRows/2) - 1);
 Nc = ifftshift(-fix(numberOfColumns/2):ceil(numberOfColumns/2) - 1);
 [Nc, Nr] = meshgrid(Nc, Nr);
 
 adjustedImage = adjustedImage.*exp(2i*pi*(-rowShift*Nr/numberOfRows - columnShift*Nc/numberOfColumns));
 adjustedImage = adjustedImage*exp(1i*phaseDifference);
 
 adjustedImage = abs(ifft2(adjustedImage));
 
 % adjust values just in case
 originalMinimum = double(min(frame(:)));
 originalMaximum = double(max(frame(:)));
 adjustedMinimum = min(adjustedImage(:));
 adjustedMaximum = max(adjustedImage(:));
 
 adjustedImage = uint16((adjustedImage - adjustedMinimum)/(adjustedMaximum - adjustedMinimum)*(originalMaximum - originalMinimum) + originalMinimum);
frame=adjustedImage;
      
 end
 previousframe=frame;
 
 
 
 frame=im2double(frame);
%  divisionfactor=prctile(prctile(frame,95),95);
%  frame=frame/divisionfactor;
 frame=log(frame);
fullstack(:,:,kk)=frame;
imagesc(frame)
% imagesc(frame,[1200 3000])

colorbar
pause(0.05)
end
%

%%
%resolution 421um*731um*5um

% sideMIP=max(fullstack,[],2);
sideMIP=prctile(fullstack,95.9,2);
sideMIP=squeeze(sideMIP);

sideMIP=sideMIP';
sideMIP=flipud(sideMIP);

bottom=prctile(reshape(sideMIP,1,size(sideMIP,1)*400),5)
top=prctile(reshape(sideMIP,1,size(sideMIP,1)*400),95)

% bottom=1200;
% top=3000;

roughsize=180;
ratio=5/(400/421)

f1=figure('Position',[40 1000  ratio*roughsize roughsize ]);
set(f1, 'PaperPositionMode', 'auto');
imagesc(sideMIP,[bottom top])


% interpolate and smooth out
smoothstack=fullstack(:,:,1);

for j=2:size(fullstack,3)
    smoothstack(:,:,2*j-1)=fullstack(:,:,j);
end
    
for j=1:size(fullstack,3)-1
    smoothstack(:,:,2*j)=mean(fullstack(:,:,[j j+1]),3);
end

smoothstack2=mean(smoothstack(:,:,[1 2]),3);
for j=2:size(smoothstack,3)-1
    smoothstack2(:,:,j)=mean(smoothstack(:,:,[j-1 j j+1]),3);
end
smoothstack2(:,:,size(smoothstack,3))=mean(smoothstack(:,:,[size(smoothstack,3)-1 size(smoothstack,3)]),3);



sideMIP=max(smoothstack2,[],2);
sideMIP=squeeze(sideMIP);
sideMIP=sideMIP';
sideMIP=flipud(sideMIP);
f1=figure('Position',[40 500  ratio*roughsize roughsize ]);
set(f1, 'PaperPositionMode', 'auto');
imagesc(sideMIP,[bottom top])

%%
f1=figure('Position',[40 500  ratio*roughsize roughsize ]);
set(f1, 'PaperPositionMode', 'auto');
imagesc(log(sideMIP))

%%


%% substacl
%resolution 421um*731um*5um

substack=fullstack(:,200:400,:);
% sideMIP=max(fullstack,[],2);
sideMIP=prctile(substack,95.9,2);
sideMIP=squeeze(sideMIP);

sideMIP=sideMIP';
sideMIP=flipud(sideMIP);

bottom=prctile(reshape(sideMIP,1,size(sideMIP,1)*size(sideMIP,2)),5)
top=prctile(reshape(sideMIP,1,size(sideMIP,1)*size(sideMIP,2)),95)

% bottom=1200;
% top=3000;

roughsize=180;
ratio=(size(sideMIP,2)/400)*5/(400/421)

f1=figure('Position',[40 1000  ratio*roughsize roughsize ]);
set(f1, 'PaperPositionMode', 'auto');
imagesc(sideMIP,[bottom top])



% interpolate and smooth out
smoothstack=substack(:,:,1);

for j=2:size(substack,3)
    smoothstack(:,:,2*j-1)=substack(:,:,j);
end
    
for j=1:size(fullstack,3)-1
    smoothstack(:,:,2*j)=mean(substack(:,:,[j j+1]),3);
end

smoothstack2=mean(smoothstack(:,:,[1 2]),3);
for j=2:size(smoothstack,3)-1
    smoothstack2(:,:,j)=mean(smoothstack(:,:,[j-1 j j+1]),3);
end
smoothstack2(:,:,size(smoothstack,3))=mean(smoothstack(:,:,[size(smoothstack,3)-1 size(smoothstack,3)]),3);



sideMIP=max(smoothstack2,[],2);
sideMIP=squeeze(sideMIP);
sideMIP=sideMIP';
sideMIP=flipud(sideMIP);
f1=figure('Position',[40 500  ratio*roughsize roughsize ]);
set(f1, 'PaperPositionMode', 'auto');
imagesc(sideMIP,[bottom top])
