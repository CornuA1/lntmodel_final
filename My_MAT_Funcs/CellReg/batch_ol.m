function batch_ol(mouse)
mouse_end = strcat('D:\Lukas\data\animals_raw\',mouse,'\2019');

for s=1:150
file_names={strcat(mouse_end,int2str(1110+s),'\adjusted_cell_bodies.mat') ,...
            strcat(mouse_end,int2str(1110+s),'_ol\adjusted_cell_bodies.mat')};
        try
            status = strcat('2019',int2str(1110+s));
            cellReg_run_ol(mouse,status,file_names)
        catch
            file_names={strcat(mouse_end,int2str(1110+s),'\prunned_cell_bodies.mat') ,...
            strcat(mouse_end,int2str(1110+s),'_ol\adjusted_cell_bodies.mat')};
            try
                status = strcat('2019',int2str(1110+s));
                cellReg_run_ol(mouse,status,file_names)
            catch
                continue
            end
        end
end

end