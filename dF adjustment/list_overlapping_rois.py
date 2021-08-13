"""
List ROIs that are classified as task engaged in VR as well as openloop

"""

import matplotlib.pyplot as plt
import numpy as np
import warnings; warnings.simplefilter('ignore')
import yaml
import json
import seaborn as sns
sns.set_style('white')
import os
with open('./loc_settings.yaml', 'r') as f:
    content = yaml.load(f)

figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop',87],['LF170222_1','Day20170615','Day20170615_openloop',96],
['LF170420_1','Day20170719','Day20170719_openloop',95],['LF170421_2','Day20170719','Day20170719_openloop',68],['LF170421_2','Day20170720','Day20170720_openloop',45],['LF170613_1','Day201784','Day201784_openloop',77]]

figure_datasets = [['LF170110_2','Day20170331','Day20170331_openloop',87]]

for r in figure_datasets:

    print(r[0], ' ', r[1])
    mouse = r[0]
    session = r[1]
    ol_session = r[2]
    tot_rois = r[3]

    with open(content['figure_output_path'] + mouse+session + os.sep + 'roi_classification.json') as f:
        roi_classification = json.load(f)

    with open(content['figure_output_path'] + mouse+ol_session + os.sep + 'roi_classification.json') as f:
        roi_classification_openloop = json.load(f)

    landmark = np.union1d(roi_classification['task_engaged_short'],roi_classification['task_engaged_long'])
    landmark_openloop = np.union1d(roi_classification_openloop['task_engaged_short'],roi_classification_openloop['task_engaged_long'])
    print(landmark)
    # print(landmark_openloop)
    # print(np.intersect1d(landmark, landmark_openloop))
