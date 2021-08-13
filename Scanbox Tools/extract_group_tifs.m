function extract_group_tifs()
[sbxName, sbxPath] = uigetfile('.sbx', 'Please select file containing imaging data.');
sbxName = strtok(sbxName, '.');
Info = sbxInfo([sbxPath, sbxName]);
start_f = 0;
end_f = Info.maxIndex;
curr = 0;
while (start_f <= end_f)
    int_f = start_f + 2500;
    if (int_f >= end_f)
        int_f = end_f+1;
    end
    k = start_f;
    nn = [sbxName, '_', int2str(curr), '.tif'];
    disp(nn);
    while (k<int_f)
        frame = sbxRead(Info, k);
        if (k == start_f)
        imwrite(frame,[sbxPath nn],'tif');
        else
        imwrite(frame,[sbxPath nn],'tif','writemode','append');
        end
        k = k + 1;
    end
    curr = curr + 1;
    start_f = start_f + 2500;
end
end