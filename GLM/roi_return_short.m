function roi_return = roi_return_short(behav_ds, roi_gcamp)

animal_location_short_idx = behav_ds(:,5) == 3; % behav_ds[:,4] in python
roi_return = roi_gcamp(animal_location_short_idx);

end