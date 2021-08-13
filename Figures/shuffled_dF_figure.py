"""Perform a (spatial) shuffle test on ROI traces to categorize ROIs according to trial location landmarks."""

import os
import numpy
import seaborn
from ruamel import yaml
from scipy import stats
from matplotlib import pyplot
from filter_trials import filter_trials

def shuffled_dF_figure(behavioral_data, dF_data, group, mouse, day, ROI, figure_format = 'png'):

    bin_size = 5.0
    number_of_shuffles = 1000
    
    track_types = ['short', 'long']
    
    track_info = {track_type: {} for track_type in track_types}
    track_trials = {track_type: [] for track_type in track_types}
    categories = {track_type: {'pre_landmark': False, 'landmark': False, 'path_integration': False, 'reward': False} for track_type in track_types}
    
    track_info['short']['track_number'] = 3
    track_info['short']['landmark_zone'] = [200.0, 240.0]
    track_info['short']['reward_zone'] = [320.0, 340.0]
    track_info['short']['track_length'] = 340.0 + bin_size
    
    track_info['long']['track_number'] = 4
    track_info['long']['landmark_zone'] = [200.0, 240.0]
    track_info['long']['reward_zone'] = [380.0, 400.0]
    track_info['long']['track_length'] = 400.0 + bin_size
        
    figure_name = 'shuffled_dF - ' + mouse + ', ' + day + ', ' + 'ROI ' + str(ROI)
    
    figure = pyplot.figure(figsize = (16, 10))
    
    # minimize empty figure space
    figure.subplots_adjust(bottom = 0.075, left = 0.05, right = 0.975, wspace = 0.05, hspace = 0.15)

    seaborn.set_style('white')
    
    # this file contains machine-specific info
    try:
        with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
            local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
    except OSError:
        print('        Could not read local settings .yaml file.')
        return
    
    # check to make sure we're only using the dF data we need
    if len(dF_data.shape) > 1:
        dF_data = dF_data[:, ROI]
    
    temp = 0
    
    for track_type in track_types:
        try:
            track_trials[track_type] = filter_trials(behavioral_data, [], ['tracknumber', track_info[track_type]['track_number']])
        except IndexError:
            print('        Unexpected track number format.')
            return
       
        temp += len(track_trials[track_type])
    
    if temp == 0:
        print('        No trials of specified types detected.')
        return
    
    task_engaged = False
    y_min = 0.0
    y_max = 0.0
    
    comparisons = {track_type: [] for track_type in track_types}
    robustness = {track_type: [] for track_type in track_types}
    average_robustness = {track_type: [] for track_type in track_types}
    
    for t_t, track_type in enumerate(track_types):
        landmark_zone = track_info[track_type]['landmark_zone']
        reward_zone = track_info[track_type]['reward_zone']
        track_length = track_info[track_type]['track_length']
        
        trials = track_trials[track_type]
        
        normal_plot = pyplot.subplot(2, len(track_types), t_t + 1)
        shuffled_plot = pyplot.subplot(2, len(track_types), t_t + 3)
        
        # mark key zones on the tracks
        normal_plot.axvspan(landmark_zone[0], landmark_zone[1], color = '0.9', zorder = 1)
        normal_plot.axvspan(reward_zone[0], reward_zone[1], color = '#D2F2FF', zorder = 1)
        
        shuffled_plot.axvspan(landmark_zone[0], landmark_zone[1], color = '0.9', zorder = 1)
        shuffled_plot.axvspan(reward_zone[0], reward_zone[1], color = '#D2F2FF', zorder = 1)
    
        N = int(track_length/bin_size)
                
        distances = numpy.linspace(bin_size/2.0, track_length - bin_size/2.0, N)
        mean_dF = numpy.empty(N)
        mean_dF.fill(numpy.NaN)
        
        dF = numpy.empty((len(trials), N))
        dF.fill(numpy.NaN)
    
        # pull out current trial and corresponding dF data and bin it   
        for t, trial in enumerate(trials):
            current_trial_location = behavioral_data[behavioral_data[:, 6] == trial, 1]
            current_trial_dF = dF_data[behavioral_data[:, 6] == trial]    
            current_trial_start = int(current_trial_location[0]/bin_size)
            current_trial_end = int(current_trial_location[-1]/bin_size)
                
            # ensure bins make logical sense
            while current_trial_end < current_trial_start:
                current_trial_location = current_trial_location[:-1]
                current_trial_dF = current_trial_dF[:-1]
                current_trial_end = int(current_trial_location[-1]/bin_size)
                
            # skip trials that are too short, where data ends before the reward zone, or where data extends beyond the track length
            if current_trial_location[-1] < reward_zone[0] or current_trial_location[-1] - current_trial_location[0] < 100.0 or current_trial_end > N:
                continue
                
            # map the data onto a matrix containing the spatial bins and trials
            dF_trial = stats.binned_statistic(current_trial_location, current_trial_dF, 'mean', (current_trial_end - current_trial_start) + 1, (current_trial_start*bin_size, (current_trial_end + 1)*bin_size))[0]
            dF[t, current_trial_start:current_trial_end + 1] = dF_trial
            normal_plot.plot(distances[current_trial_start:current_trial_end + 1], dF_trial, c = '0.8')
            y_min = numpy.nanmin([y_min, numpy.amin(dF_trial)])
            y_max = numpy.nanmax([y_max, numpy.amax(dF_trial)])
            
        # exclude trials with no dF data
        trials = trials[~numpy.isnan(dF).all(1)]
        dF = dF[~numpy.isnan(dF).all(1), :]
        
        for spatial_bin in range(N):
            number_of_nans = numpy.sum(numpy.isnan(dF[:, spatial_bin]))
            
            if number_of_nans/len(trials) < 0.2:
                if number_of_nans/len(trials) > 0.05:
                    print('Warning: More than 5% of trials have NaNs in spatial bin ' + str(spatial_bin))

                mean_dF[spatial_bin] = numpy.nanmean(dF[:, spatial_bin])
            else:
                print('Warning: More than 20% of trials have NaNs in spatial bin ' + str(spatial_bin))
        
#        # plot means, excluding spatial bins with NaNs
#        indices = ~numpy.isnan(dF).any(0)
#        
#        distances[~indices] = numpy.nan
#        
#        mean_dF[indices] = dF[:, indices].mean(axis = 0)
#        mean_dF[~indices] = numpy.nan
        
        normal_plot.plot(distances, mean_dF, c = seaborn.xkcd_rgb['windows blue'], lw = 3)
    
        shuffled_dF = numpy.empty((number_of_shuffles, dF.shape[0], N))
        shuffled_dF.fill(numpy.NaN)
    
        # now shuffle dF data around on a per trial basis
        for n in range(number_of_shuffles):
            shuffled_dF[n, :, :] = dF
            
            # data is only shuffled within spatial bins that previously contained data
            for t in range(len(trials)):
                trial_indices = ~numpy.isnan(dF[t, :])
                shuffled_dF[n, t, trial_indices] = numpy.roll(dF[t, trial_indices], numpy.random.randint(0, len(dF[t, trial_indices])))
    
            # same as for the regular data, ignore spatial bins with fewer than 10 non-NaN values
            for spatial_bin in range(N):
                if numpy.sum(~numpy.isnan(shuffled_dF[n, :,  spatial_bin])) < 10:
                    shuffled_dF[n, :, spatial_bin] = numpy.NaN
        
#            shuffled_plot.plot(distances[indices], shuffled_dF[n, :, indices].mean(axis = 1), c = '0.8', lw = 3)
            shuffled_plot.plot(distances, numpy.nanmean(shuffled_dF[n, :, :], axis = 0), c = '0.8', lw = 3)

        # average across trials
#        mean_shuffled_dF = numpy.empty((number_of_shuffles, N))
#        mean_shuffled_dF[:, indices] = shuffled_dF[:, :, indices].mean(axis = 1)
#        mean_shuffled_dF[:, ~indices] = numpy.nan
        mean_shuffled_dF = numpy.nanmean(shuffled_dF, 1)

        # find the standard deviation at each bin
        std_shuffled_dF = mean_shuffled_dF.std(axis = 0)

        # average across shuffles for each spatial bin
        mean_mean_shuffled_dF = mean_shuffled_dF.mean(axis = 0)

        threshold = mean_mean_shuffled_dF + 2.0*std_shuffled_dF
    
        # compare to threshold at each spatial bin
        comparisons[track_type] = mean_dF - threshold
    
        consecutive = 0
        
        robustness[track_type] = dF > threshold
        
        robustness[track_type] = numpy.sum(robustness[track_type], axis = 0)/len(trials)
        
        average_robustness[track_type] = numpy.mean(robustness[track_type])
        
        # only consider a ROI where at least one spatial bin with transients above "threshold" in more than 50% of trials
        if any(robustness[track_type] >= 0.25):
            
            # check to see if there are at least 3 consecutive spatial bins over threshold
            for c, comparison in enumerate(comparisons[track_type]):
                if comparison != numpy.nan:
                    if comparison > 0.0:
                        if consecutive == 0:
                            first = c
                            
                        consecutive += 1
                    else:
                        if consecutive >= 3:
                            temp = distances[first:c].mean()
                    
                            if 0.0 < temp <= landmark_zone[0]:
                                categories[track_type]['pre_landmark'] = True
                            if landmark_zone[0] < temp <= landmark_zone[1]:
                                categories[track_type]['landmark'] = True
                            if landmark_zone[1] < temp <= reward_zone[0]:
                                categories[track_type]['path_integration'] = True
                            if reward_zone[0] < temp:
                                categories[track_type]['reward'] = True
                                
                        consecutive = 0
            
            # check for ROI activity at the end of the tracks
            if consecutive >= 3:
                temp = distances[first:c].mean()
                
                if 0.0 < temp <= landmark_zone[0]:
                    categories[track_type]['pre_landmark'] = True
                if landmark_zone[0] < temp <= landmark_zone[1]:
                    categories[track_type]['landmark'] = True
                if landmark_zone[1] < temp <= reward_zone[0]:
                    categories[track_type]['path_integration'] = True
                if reward_zone[0] < temp:
                    categories[track_type]['reward'] = True

        if any(categories[track_type].values()):
            normal_plot.set_title(track_type + '\n*******', fontsize = 20)
            task_engaged = True
        else:
            normal_plot.set_title(track_type, fontsize = 20)
            
        normal_plot.plot(distances, threshold, color = 'k', linestyle = 'dashed')
        normal_plot.set_xlim([0, track_length])
        
        shuffled_plot.plot(distances, mean_dF, c = seaborn.xkcd_rgb['windows blue'], lw = 3)
        shuffled_plot.plot(distances, threshold, color = 'k', linestyle = 'dashed')
        shuffled_plot.set_xlim([0, track_length])
        shuffled_plot.set_xlabel('location (cm)', fontsize = 20)
        
        if t_t == 0:
            normal_plot.set_ylabel('dF/F', fontsize = 20)
            shuffled_plot.set_ylabel('dF/F', fontsize = 20)

    if task_engaged:
        figure.suptitle(figure_name + '\ntask engaged ROI', wrap = True, fontsize = 20)
    else:
        figure.suptitle(figure_name + '\nnot task engaged ROI', wrap = True, fontsize = 20)
        
    axes = figure.get_axes()

    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.tick_params(reset = 'on', axis= 'both', direction = 'in', length = 4, right = 'off', top='off')
            
        # set the y-axis limits according to the maximum dF value in either track type, for comparison
        axis.set_ylim([y_min, y_max])
    
    figure_path = local_settings['figure_output_path']
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
        
    figure_path += os.sep + mouse
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
        
    figure_path += os.sep + 'shuffled dF'
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
    
    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)
    
    # close the figure to save memory
    pyplot.close(figure)
    
    # add current ROI to .yaml file if it is task engaged
    if task_engaged:
        try:
            with open(local_settings['yaml_file'], 'r') as yaml_file:
                project_yaml = yaml.round_trip_load(yaml_file, preserve_quotes = True)
        except OSError:
            print('        Could not read project .yaml file.')
            return comparisons, robustness
            
        # the yaml file nests data as dictionaries within lists
        modified_yaml = project_yaml.copy()
        
        modified_yaml = modified_yaml[group]
        
        for m in range(len(modified_yaml)):
            if mouse in modified_yaml[m]:
                modified_yaml = modified_yaml[m]
                modified_yaml = modified_yaml[mouse]
                break
        
        try:
            modified_yaml[0]['date']
        except KeyError:
            print('        No session information in project .yaml file.')
            return comparisons, robustness
        except TypeError:
            print('        No session information in project .yaml file.')
            return comparisons, robustness
        
        temp = int(day.replace('Day', '').replace('_openloop', '').replace('_gratings', ''))
        
        for d in range(len(modified_yaml)):
            if temp == modified_yaml[d]['date']:
                if 'openloop' in day:
                    if modified_yaml[d]['rectype'] == 'OPENLOOP':
                        break
                elif 'gratings' in day:
                    if modified_yaml[d]['rectype'] == 'GRATINGS':
                        break
                else:
                    break
            elif d == len(modified_yaml) - 1:
                print('        Day not found in project .yaml file.')
                return comparisons, robustness
            
        modified_yaml = modified_yaml[d]
            
        if 'task_engaged' in modified_yaml:
            temp = modified_yaml['task_engaged']
            
            if ROI not in temp:
                temp.append(ROI)
                temp.sort()
                temp = yaml.comments.CommentedSeq(temp)
                temp.fa.set_flow_style()
                modified_yaml['task_engaged'] = temp
        else:
            temp = yaml.comments.CommentedSeq([ROI])
            temp.fa.set_flow_style()
            modified_yaml['task_engaged'] = temp
            
        for track_type in track_types:
            if any(categories[track_type].values()):
                if track_type not in modified_yaml:
                    modified_yaml[track_type] = [{}]
                    
                keys = []
                    
                for item in categories[track_type]:
                    if categories[track_type][item]:
                        keys.append(item)
                    
                # lists of ROIs need to be specially formatted to look pretty in the actual file
                for key in keys:
                    if key in modified_yaml[track_type][0]:
                        temp = modified_yaml[track_type][0][key]
                        
                        if ROI not in temp:
                            temp.append(ROI)
                            temp.sort()
                            temp = yaml.comments.CommentedSeq(temp)
                            temp.fa.set_flow_style()
                            modified_yaml[track_type][0][key] = temp
                    else:
                        temp = yaml.comments.CommentedSeq([ROI])
                        temp.fa.set_flow_style()
                        modified_yaml[track_type][0][key] = temp
                
        project_yaml[group][m][mouse][d] = modified_yaml
                    
        try:
            with open(local_settings['yaml_file'], 'w') as yaml_file:
                yaml.round_trip_dump(project_yaml, yaml_file, default_flow_style = False)
        except OSError:
            print('        Could not modify project .yaml file.')
            return comparisons, robustness
        
        return comparisons, robustness