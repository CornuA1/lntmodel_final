"""
print the content of dictionaries for individual session_rois

"""

import json

def print_dict(json_path):
    with open(json_path, 'r') as f:
        sess_dict = json.load(f)
        print(len(sess_dict['valid_rois']))
        print(len(sess_dict['roi_numbers']))
        print(len(sess_dict['norm_value']))
        print(len(sess_dict['transient_AUC_mean']))
        # print(sess_dict.keys())

        # for keys in sess_dict:
        #     print(keys)
            # if type(sess_dict[keys]) is list:
            #     print(len(sess_dict[keys]))
            # else:
            #     print(sess_dict[keys])
            # print(type(sess_dict[keys]))
        # print(sess_dict['roi_numbers'])
        # print(sess_dict['valid_rois'])
        # print(sess_dict['invalid_rois'])



if __name__ == "__main__":
    print_dict('E:/MTH3_figures/LF170110_2/LF170110_2_Day201748_1.json')
