"""
Search project configuration file (.yaml) by certain criteria (such as session type) and retrieve all matching recording sessions.

Parameters
---
path :  string
        path to project configuration file

group : string
        which group of animals (can be multiple)

filterprops :   tuples
                filter properties. Can be multiple filter properties

Outputs
---
[mousename, date, recoding type, task] :
                    return mousename and dataset (typically the date of a recording)
                    matching the filtering criteria

"""


import yaml
import numpy as np


def searchyaml(path, group, filterprops):
    keys_found = np.zeros(len(filterprops))
    mice_list = []
    date_list = []
    rectype_list = []
    task_list = []
    task_engaged = []
    shortrois = []
    longrois = []

    with open(path, 'r') as f:
        content = yaml.load(f)

    # Iterate through groups/mice/dataset info to find dataset
    for gr in group:
        for e in content[gr]:
            for s in list(e):
                if type(e[s]) == list:
                    for x in e[s]:
                        for i, y in enumerate(filterprops):
                            try:
                                if x[y[0]] == y[1]:
                                    keys_found[i] = 1
                            except KeyError:
                                if y[0] == 'mouse':
                                    if s == y[1]:
                                        keys_found[i] = 1
                                else:
                                    raise KeyError('Keyword not found')

                        if np.sum(keys_found) == len(filterprops):
                            mice_list.append(s)
                            date_list.append(x['date'])
                            rectype_list.append(x['rectype'])
                            task_list.append(x['task'])
                            if 'task_engaged' in x:
                                task_engaged.append(x['task_engaged'])
                            if 'short' in x:
                                shortrois.append(x['short'])
                            if 'long' in x:
                                longrois.append(x['long'])
                        keys_found = np.zeros(len(filterprops))
    return [mice_list, date_list, rectype_list, task_list, task_engaged, shortrois, longrois]
