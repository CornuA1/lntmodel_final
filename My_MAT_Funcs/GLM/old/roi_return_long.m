function roi_return = roi_return_long(behav_ds, roi_gcamp)

animal_location_short_idx = behav_ds(:,5) == 4; % behav_ds[:,4] in python
roi_return = roi_gcamp(animal_location_short_idx);

end