clear;

loc_info = ReadYaml("C:\Users\Lou\Documents\repos\LNT\loc_settings.yaml");
MOUSE = 'LF191022_1';
SESSION = ['20191115';'20191125';'20191204';'20191209'];
numb = size(SESSION,1);

for i = 1:size(SESSION,1)
    shot_mat = [];
    long_mat = [];
    for x = 1:100
        file = strcat(loc_info.raw_dir(2:end-1), MOUSE, '\', SESSION(i,:), '\data_glm\',int2str(x),'.mat');
        data = load(file);
        if x == 1
            shot_mat = data.coef_short(2:end);
            long_mat = data.coef_long(2:end);
        else
            shot_mat = [shot_mat, data.coef_short(2:end)];
            long_mat = [long_mat, data.coef_long(2:end)];
        end
    end
    short_mean = mean(shot_mat,2);
    makeFigure2(1,1,short_mean,1,1);
    long_mean = mean(long_mat,2);
    makeFigure2(1,1,long_mean,1,1);
end