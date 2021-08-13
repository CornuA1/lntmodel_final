function create_match_file(mouse,session,naiveorexpert)
 
cellreg_script = strcat('D:\Lukas\data\animals_raw\',mouse,'\results_',naiveorexpert,'\cellRegistered_',mouse,'_',naiveorexpert,'.mat');
cell_index_lib = load(cellreg_script);
cellind = cell_index_lib.cell_registered_struct.cell_to_index_map;
 
final_map = zeros(1,2);
 
for i=1:length(cellind)
    if cellind(i,1) ~= 0 && cellind(i,2) ~= 0
        final_map = cat(1,final_map,cellind(i,:));
    end
end
new_res = final_map(2:end,:);
save(strcat('D:\Lukas\data\animals_raw\',mouse,'\',session,'\matched_cells.mat'), 'new_res')
end