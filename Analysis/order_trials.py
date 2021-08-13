"""
Order and return trials based on user-specified criteria

Parameters
-------
raw_behav : ndarray
            raw behaviour dataset

trial_list : list
            list of trials to be ordered. This function requires a list of trials
            to minimize assumptions about which trials are to be ordered in
            a given session (e.g. we don't want blackbox trials to be ordered)

orderprops : tuple
            provide information about how trials should be ordered. Since
            different order types may need different parameters, this function
            argument is intentionally flexible

pre_select : ndarray
            trialnumbers that have been pre-filtered. Output will be a
            subset of this input.

Outputs
-------
ordered_trials : ndarray
            trialnumbers, ordered by user-specified criteria

"""

import numpy as np
from event_ind import event_ind

def time_between_points_ordered(raw_behav, trial_list, orderprops):
    orderpoint_1 = orderprops[1]
    orderpoint_2 = orderprops[2]
    ordered_trials = np.zeros((len(trial_list),2))
    events_short_lmcenter_all = event_ind(raw_behav, orderpoint_1)
    events_short_reward_all = event_ind(raw_behav,orderpoint_2)
    for i,tr in enumerate(trial_list):
        events_short_lmcenter = events_short_lmcenter_all[events_short_lmcenter_all[:,1] == tr,:]
        events_short_reward = events_short_reward_all[np.in1d(events_short_reward_all[:,1],tr),:]
        # events_short_lmcenter = event_ind(raw_behav, orderpoint_1, [tr])
        # events_short_reward = event_ind(raw_behav,orderpoint_2, [tr])
        if events_short_lmcenter.shape[0] > 0 and events_short_reward.shape[0] > 0:
            ordered_trials[i,0] = raw_behav[events_short_reward[0,0].astype(int),0] - raw_behav[events_short_lmcenter[0,0].astype(int),0]
            ordered_trials[i,1] = events_short_reward[0,1]
        else:
            ordered_trials = np.delete(ordered_trials, i, 0)
    # order trials by time that has passed between points (lowest to highest)
    ordered_trials = np.flipud(ordered_trials[ordered_trials[:,0].argsort()])
    return ordered_trials

def order_trials( raw_behav, trial_list, orderprops=[]):
    """ call orderfunction as specified in orderprops """
    # call desired ordered function
    print(orderprops[0])
    if orderprops[0] is 'time_between_points':
        ordered_trials = time_between_points_ordered(raw_behav, trial_list, orderprops)
    else:
        raise ValueError('Order type not recognised.')

    # return trial numbers that have passed the ordered and exist in pre_select
    return ordered_trials
