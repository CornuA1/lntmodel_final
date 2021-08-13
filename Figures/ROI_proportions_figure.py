"""Compare session task scores against proportions of ROIs of different categories."""

import os
import numpy
import seaborn
from ruamel import yaml
from matplotlib import pyplot

def ROI_proportions_figure(group, mouse, figure_format = 'png'):
    
    track_types = ['short', 'long']
    categories = ['pre_landmark', 'landmark', 'path_integration', 'reward']
    colors = ['crimson', 'indigo', 'skyblue', 'k']
    
    width = 2.0
                    
    figure_name = 'ROI_proportions - ' + mouse
    
    figure = pyplot.figure(figsize = (16, 10))
    
    seaborn.set_style('white')
    
    # this file contains machine-specific info
    try:
        with open('..' + os.sep + 'loc_settings.yaml', 'r') as yaml_file:
            local_settings = yaml.load(yaml_file, Loader = yaml.Loader)
    except OSError:
        print('        Could not read local settings .yaml file.')
        return
      
    try:
        with open('..' + os.sep + 'detailed_summary.yaml', 'r') as yaml_file:
            summary_yaml = yaml.round_trip_load(yaml_file, preserve_quotes = True)
    except OSError:
        print('        Could not read summary .yaml file.')
        return
    
    # the yaml file nests data as dictionaries within lists    
    try:
        summary_yaml = summary_yaml[group]
    except KeyError:
        return
    
    mouse_not_found = True    
    
    for m in range(len(summary_yaml)):
        if mouse in summary_yaml[m]:
            summary_yaml = summary_yaml[m]
            summary_yaml = summary_yaml[mouse]
            
            mouse_not_found = False            
            break
        
    if mouse_not_found:
        return        
        
    dates = range(len(summary_yaml))        
        
    ROI_categories = [{track_type: {category: 0 for category in categories} for track_type in track_types} for d in dates]
        
    task_scores = [[] for d in dates]
    
    first_task_score = True

    for d in dates:
        ROI_categories[d]['total'] = 0
        
        for date in summary_yaml[d]:
            try:
                ROI_categories[d]['total'] = summary_yaml[d][date]['number_of_ROIs']['total']
            except KeyError:
                continue
            
            try:
                ROI_categories[d]['task_engaged'] = summary_yaml[d][date]['number_of_ROIs']['task_engaged']
            except KeyError:
                continue
                
            try:
                task_scores[d] = summary_yaml[d][date]['task_score']
                
                if first_task_score:
                    min_task_score = task_scores[d]
                    max_task_score = task_scores[d]
                    
                    first_task_score = False
                else:
                    min_task_score = numpy.amin([min_task_score, task_scores[d]])
                    max_task_score = numpy.amax([max_task_score, task_scores[d]])
            except KeyError:
                continue
            
            bottom = 0.0            
            
            for track_type in track_types:
                for c, category in enumerate(categories):
                    try:
                        ROI_categories[d][track_type][category] = summary_yaml[d][date]['number_of_ROIs'][track_type + '_track'][category]
                    except KeyError:
                        continue
                    
                    proportion = ROI_categories[d][track_type][category]/ROI_categories[d]['total']*100.0
        
                    pyplot.bar(task_scores[d] - width/2.0, proportion, width = width, color = colors[c], bottom = bottom, label = str(track_type) + ': ' + str(category))
                    
                    bottom += proportion
                    
            pyplot.legend(fontsize = 20)
            
    if first_task_score:
        return
            
    pyplot.set_xlim([min_task_score - 2.0*width, max_task_score + 2.0*width])            
    pyplot.set_ylim([0.0, 100.0])    
        
    pyplot.set_xlabel('task score', fontsize = 20)
    pyplot.set_ylabel('% of total ROIs', fontsize = 20)
    pyplot.set_title(figure_name, fontsize = 20)
        
    pyplot.spines['top'].set_visible(False)
    pyplot.spines['right'].set_visible(False)
    pyplot.tick_params(reset = 'on', axis = 'both', direction = 'in', length = 4, right = 'off', top='off')
    
    figure_path = local_settings['figure_output_path']
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
        
    figure_path += os.sep + mouse
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
        
    figure_path += os.sep + 'ROI proportions'
    
    if not os.path.isdir(figure_path):
        os.mkdir(figure_path)
    
    figure.savefig(figure_path + os.sep + figure_name + '.' + figure_format, format = figure_format)
    
    # close the figure to save memory
    pyplot.close(figure)