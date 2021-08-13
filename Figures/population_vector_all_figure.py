"""Spatially bin dF/F data and calculate neural population vectors."""

import os
import h5py
import numpy
import seaborn
from ruamel import yaml
from scipy import stats
from matplotlib import pyplot
from filter_trials import filter_trials

def population_vector_all_figure(mice, figure_name, figure_format = 'png'):
    
    def calculate_population_vectors(behavioral_data, dF_data, ROIs):
        
        distances = {track_type: [] for track_type in track_types}
        population_vectors = {track_type: [] for track_type in track_types}
    
        for t_t, track_type in enumerate(track_types):
            track_number = track_info[track_type]['track_number']
            reward_zone = track_info[track_type]['reward_zone']
            track_length = track_info[track_type]['track_length']
            
            try:
                trials = filter_trials(behavioral_data, [], ['tracknumber', track_number])
            except IndexError:
                print('        Unexpected track number format.')
                return
            
            N = int(track_length/bin_size)
            
            distances[track_type] = numpy.linspace(bin_size/2.0, track_length - bin_size/2.0, N)
            
            population_vectors[track_type] = numpy.full((len(ROIs), N), numpy.NaN)
            
            for r, ROI in enumerate(ROIs):
                dF = numpy.full((len(trials), N), numpy.NaN)
            
                # pull out current trial and corresponding dF data and bin it   
                for t, trial in enumerate(trials):
                    current_trial_location = behavioral_data[behavioral_data[:, 6] == trial, 1]
                    current_trial_dF = dF_data[behavioral_data[:, 6] == trial, :]    
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
                    dF_trial = stats.binned_statistic(current_trial_location, current_trial_dF[:, ROI], 'mean', (current_trial_end - current_trial_start) + 1, (current_trial_start*bin_size, (current_trial_end + 1)*bin_size))[0]
                    dF[t, current_trial_start:current_trial_end + 1] = dF_trial
                    
                # exclude trials with no dF data
                dF = dF[~numpy.isnan(dF).all(1), :]
            
                # normalize the data for each ROI
                mean_dF = numpy.nanmean(dF, axis = 0)
                population_vectors[track_type][r, :] = (mean_dF - mean_dF.min())/(mean_dF.max() - mean_dF.min())
            
        return distances, population_vectors
    
    bin_size = 5.0
    
    track_types = ['short', 'long']
    
    track_info = {track_type: {} for track_type in track_types}
    analysis = {track_type: {'distances': [], 'population_vector': []} for track_type in track_types}
    plots = {track_type: [] for track_type in track_types}
    
    track_info['short']['track_number'] = 3
    track_info['short']['landmark_zone'] = [200.0, 240.0]
    track_info['short']['reward_zone'] = [320.0, 340.0]
    track_info['short']['track_length'] = 340.0 + bin_size
    
    track_info['long']['track_number'] = 4
    track_info['long']['landmark_zone'] = [200.0, 240.0]
    track_info['long']['reward_zone'] = [380.0, 400.0]
    track_info['long']['track_length'] = 400.0 + bin_size
    
    figure = pyplot.figure(figsize = (16, 10))
    
    # minimize empty figure space
    figure.subplots_adjust(bottom = 0.05, left = 0.05, right = 0.975, wspace = 0.2, hspace = 0.25)
    
    RdBu_r = pyplot.get_cmap('RdBu_r')

    seaborn.set_style('white')
    
    # this file contains machine-specific info
    try:
        with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
            local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
    except OSError:
        print('        Could not read local settings .yaml file.')
        return
        
    for mouse in mice:        
        print(mouse['name'])
        
        try:
            HDF5_data = h5py.File(local_settings['imaging_dir'] + mouse['name'] + os.sep + mouse['name'] + '.h5', 'r')
        except OSError:
            print('    No HDF5 file.')
            continue
        
        days = [day for day in HDF5_data]
        
        for day in days:
            if 'gratings' not in day and 'openloop' not in day:
                print('    ' + day)
                
                try:
                    with open(local_settings['active_yaml_file'], 'r') as yaml_file:
                        project_yaml = yaml.load(yaml_file, Loader = yaml.Loader)
                except OSError:
                    print('        Could not read project .yaml file.')
                    return
                
                # the yaml file nests data as dictionaries within lists
                project_yaml = project_yaml[mouse['group']]
                
                for m in range(len(project_yaml)):
                    if mouse['name'] in project_yaml[m]:
                        project_yaml = project_yaml[m]
                        project_yaml = project_yaml[mouse['name']]
                        break
                        
                try:
                    project_yaml[0]['date']
                except KeyError:
                    print('        No session information in project .yaml file.')
                    return
                except TypeError:
                    print('        No session information in project .yaml file.')
                    return
                
                try:
                    behavioral_data = numpy.copy(HDF5_data[day + '/behaviour_aligned'])
                except KeyError:
                    print('        No behaviour_aligned.')
                    continue
                except OSError:
                    print('        Corrupted behaviour_aligned.')
                    continue
                
                try:
                    dF_data = numpy.copy(HDF5_data[day + '/dF_win'])
                except KeyError:
                    print('        No dF_win.')
                    continue
        
                if dF_data.shape[1] == 1:
                    print('        Only one ROI detected.')
                    continue
            
                try:
                    short_trials = filter_trials(behavioral_data, [], ['tracknumber', 3])
                    long_trials = filter_trials(behavioral_data, [], ['tracknumber', 4])
                except IndexError:
                    print('        Unexpected track number format.')
                    continue
                
                if len(short_trials) == 0:
                    print('        No short trials detected.')
                    continue
                
                if len(long_trials) == 0:
                    print('        No long trials detected.')
                    continue
                
                temp = int(day.replace('Day', '').replace('_openloop', '').replace('_gratings', ''))
                
                skip = False
            
                for d in range(len(project_yaml)):
                    if temp == project_yaml[d]['date']:
                        if 'openloop' in day:
                            if project_yaml[d]['rectype'] == 'OPENLOOP':
                                break
                        elif 'gratings' in day:
                            if project_yaml[d]['rectype'] == 'GRATINGS':
                                break
                        else:
                            break
                    elif d == len(project_yaml) - 1:
                        print('        Day not found in project .yaml file.')
                        skip = True
                        continue
                    
                if skip:
                    continue
                    
                project_yaml = project_yaml[d]
            
                if 'task_engaged' in project_yaml:
                    ROIs = project_yaml['task_engaged']
                else:
                    print('        No task-engaged ROIs.')
                    continue
                
                if len(ROIs) == 1:
                    print('        Only one task-engaged ROI detected.')
                    continue
                
                distances, population_vectors = calculate_population_vectors(behavioral_data, dF_data, ROIs)
        
                for track_type in track_types:
                    if ~numpy.isnan(population_vectors[track_type]).all():
                        if len(analysis[track_type]['distances']) == 0:
                            analysis[track_type]['distances'] = distances[track_type]
                            analysis[track_type]['population_vector'] = population_vectors[track_type]
                        else:
                            analysis[track_type]['population_vector'] = numpy.concatenate((analysis[track_type]['population_vector'], population_vectors[track_type]), axis = 0)
                
    plots['correlation'] = pyplot.subplot(len(track_types), 2, 2)
    
    for t_t, track_type in enumerate(track_types):
        
        # exclude spatial bins with NaNs
        indices = ~numpy.isnan(analysis[track_type]['population_vector']).any(0)
        
        analysis[track_type]['distances'] = analysis[track_type]['distances'][indices]
        analysis[track_type]['population_vector'] = analysis[track_type]['population_vector'][:, indices]
        
        plots[track_type] = pyplot.subplot(len(track_types), 2, t_t*(t_t + 1) + 1)
            
        if t_t == 0:
        
            # arrange the data by (first) maxima of one of the track types, eg. short
            sorting = list(zip(list(range(analysis[track_type]['population_vector'].shape[0])), analysis[track_type]['population_vector'].argmax(axis = 1)))
            
            sorting.sort(key = lambda pair: pair[1])
            
            ordering_indices = [pair[0] for pair in sorting]
                
            analysis[track_type]['population_vector'] = analysis[track_type]['population_vector'][ordering_indices, :]
                
    for t_t_1, track_type_1 in enumerate(track_types):
        landmark_zone_1 = track_info[track_type_1]['landmark_zone']
        reward_zone_1 = track_info[track_type_1]['reward_zone']
        track_length_1 = track_info[track_type_1]['track_length']
        
        # pcolormesh has an annoying label strategy that needs to be compensated by offsetting the mesh, like so
        distances_1 = analysis[track_type_1]['distances']
        temp_distances_1 = distances_1.copy()
        population_vector_1 = analysis[track_type_1]['population_vector']
        
        for d in range(len(distances_1)):
            temp_distances_1[d] = temp_distances_1[d] - (distances_1[1] - distances_1[0])/2.0
        
        temp_distances_1 = numpy.append(temp_distances_1, temp_distances_1[-1] + (distances_1[1] - distances_1[0]))
        
        plot_colors = ['crimson', 'indigo', 'deepskyblue']
        
        for t_t_2, track_type_2 in enumerate(track_types):
            if t_t_2 < t_t_1:
                reward_zone_2 = track_info[track_type_2]['reward_zone']
                
                distances_2 = analysis[track_type_2]['distances']
                temp_distances_2 = distances_2.copy()
                population_vector_2 = analysis[track_type_2]['population_vector']
                
                for d in range(len(distances_2)):
                    temp_distances_2[d] = temp_distances_2[d] - (distances_2[1] - distances_2[0])/2.0
                
                temp_distances_2 = numpy.append(temp_distances_2, temp_distances_2[-1] + (distances_2[1] - distances_2[0]))
                
                x_grid, y_grid = numpy.meshgrid(temp_distances_1, temp_distances_2)
        
                diagonals = numpy.intersect1d(distances_1, distances_2)
        
                correlations = numpy.empty((diagonals.shape[0]))
                
                for i in range(diagonals.shape[0]):
                    temp_1 = population_vector_1[:, distances_1 == diagonals[i]].T
                    temp_2 = population_vector_2[:, distances_2 == diagonals[i]]
                    correlations[i] = numpy.dot(temp_1/numpy.linalg.norm(temp_1), temp_2/numpy.linalg.norm(temp_2))
                
                correlation_plot = plots['correlation']
        
                correlation_plot.plot(diagonals, correlations, color = plot_colors[t_t_2 % len(plot_colors)], linewidth = 3.0)
        
                correlation_plot.set_xlabel('location (cm)')
                correlation_plot.set_ylabel('correlation')
                correlation_plot.set_xlim(0.0, numpy.amax([track_length_1, track_info[track_type_2]['track_length']]))
                correlation_plot.set_ylim(0.0, 1.0)
                
                correlation_plot.axvspan(numpy.amin([reward_zone_1[0], reward_zone_2[0]]), numpy.amax([reward_zone_1[1], reward_zone_2[1]]), color = '#D2F2FF')
                correlation_plot.axvspan(landmark_zone_1[0], landmark_zone_1[1], color = '0.9', zorder = 1)
        
        temp_ROIs = list(range(numpy.size(population_vector_1, 1)))
        temp = temp_ROIs.copy()
        
        for r in range(len(ROIs)):
            temp_ROIs[r] = temp_ROIs[r] - (temp[1] - temp[0])/2.0
        
        temp_ROIs.append(temp_ROIs[-1] + (temp[1] - temp[0]))
        
        x_grid, y_grid = numpy.meshgrid(temp_distances_1, temp_ROIs)
        
        population_plot = plots[track_type_1]
        
        population_plot.pcolormesh(x_grid, y_grid, (population_vector_1.T/numpy.abs(population_vector_1).max(axis = 1)).T, vmin = -1.0, vmax = 1.0, cmap = RdBu_r)
        
        population_plot.set_xlabel('location (' + track_type_1 + ') (cm)')
        population_plot.set_ylabel('ROIs')
        population_plot.set_xlim(0, track_length_1)
        population_plot.set_ylim(0, len(ROIs))
        
        population_plot.axvline(landmark_zone_1[0], color = 'k')
        population_plot.axvline(landmark_zone_1[1], color = 'k')
        population_plot.axvline(reward_zone_1[0], color = 'b')
        population_plot.axvline(reward_zone_1[1], color = 'b')
    
    figure.suptitle(figure_name, wrap = True, fontsize = 25)
        
    axes = figure.get_axes()

    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.tick_params(reset = 'on', axis= 'both', direction = 'in', length = 4, right = 'off', top='off')
    
    figure_path = local_settings['figure_output_path']
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
    
    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)
    
    print(figure_path + os.sep + figure_name)
    
    # close the figure to save memory
    pyplot.close(figure)