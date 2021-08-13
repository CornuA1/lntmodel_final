%remote processing
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
for kk=10:150
    
    indexyo=index(kk,:)
zstack_hackepreanalysis3(indexyo)
end