"""
Central place to store and modify parameters used across analysis pipeline

"""

# fraction of trials a given roi has to be 'active' at
MIN_FRACTION_ACTIVE = 0.25
# minimum amplitude of the mean trace (i.e. max(mean_trace) - min(mean_trace))
MIN_MEAN_AMP = 0.2
MIN_MEAN_AMP_BOUTONS = 0.1
# min z-score that mean trace has to be above shuffled distribution
MIN_ZSCORE = 3
# min number of trials active (this is somewhat redundant with MIN_FRACTION_ACTIVE, but its here for historic reasons)
MIN_TRIALS_ACTIVE = 0.25
# I honestly do not remember right now what this is for or where it is used (perhaps its not used anymore at all?)
MIN_DF = 0.1

# fraction of trials at which the animal had to be in a certain location to be considered part of the mean
MEAN_TRACE_FRACTION = 0.5

# time window around which the peak can happen in the
PEAK_MATCH_WINDOW = 2
