function [fit, test] = split_test_and_fit_behave(stuff_to_cut, behave_data, track_num)
animal_location_idx = behave_data(:,5) == track_num;
animal_location = behave_data(animal_location_idx,2);
num_trials = 1;
vval = -10;
for x = 1:size(animal_location)
    if animal_location(x) - vval < 0
        num_trials = num_trials + 1;
    end
    vval = animal_location(x);
end
fit = [];
test = [];
if num_trials < 11
    current_trial = 1;
    vval = -10;
    for x = 1:size(animal_location)
        if current_trial == round(num_trials/2)
            test = cat(1,test,stuff_to_cut(x,:));
        else
            fit = cat(1,fit,stuff_to_cut(x,:));
        end
        if animal_location(x) - vval < 0
            current_trial = current_trial + 1;
        end
        vval = animal_location(x);
    end
else
    current_trial = 1;
    vval = -10;
    for x = 1:size(animal_location)
        if 0 == mod(current_trial,10)
            test = cat(1,test,stuff_to_cut(x,:));
        else
            fit = cat(1,fit,stuff_to_cut(x,:));
        end
        if animal_location(x) - vval < 0
            current_trial = current_trial + 1;
        end
        vval = animal_location(x);
    end    
end
end