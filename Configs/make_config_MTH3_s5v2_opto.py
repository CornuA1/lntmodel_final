"""
Create a config file with given parameters. Primarily used to get a random sequence of tracks

"""

import random

def make_config_MTH3_VD(fname, lines, tracknrs, stim_amp, mousename=[]):
    if not mousename:
        mousename = 'C57'

    with open("{0}.csv".format(fname), 'w') as config_file:
        config_file.write('Experiment;MTH3_vr2_opto;Length (min);90;Day;5;ExpGroup;1;Mouse;{0};DOB;00000000;Strain;C57;Stop Threshold;0.7;Valve Open Time;0.3;Comments;opto\n'.format(mousename))
        previous_opto_stim = False
        next_opto_stim_forced = False
        for i in range(lines):
            track = tracknrs[0]
            tracktype = random.choice(tracknrs)
            if tracktype == 3:
                t_reset = 400
                rz_start = 320
                rz_end = 340
                landmark = 240
                default = 340
            elif tracktype == 4:
                t_reset = 460
                rz_start = 380
                rz_end = 400
                landmark = 240
                default = 400
            # 1:2 chance to have the masking stimulus on
            opto_stim = random.choice([False,True])
            if opto_stim == False and previous_opto_stim == False and next_opto_stim_forced == False:
                config_file.write('track;{0};prelick;1.0;gain modulation;1.0;rewarded;1;Reset;{1};RZ start;{2}.0;RZ end;{3}.0;landmark;{4}.0;default;{5};openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n'.format(tracktype,t_reset,rz_start,rz_end,landmark,default))

            elif opto_stim == True and previous_opto_stim == False and next_opto_stim_forced == False:
                config_file.write('track;{0};prelick;1.0;gain modulation;1.0;rewarded;1;Reset;{1};RZ start;{2}.0;RZ end;{3}.0;landmark;{4}.0;default;{5};openloop;-1;bbreset;0;opto_mask;1;opto_stim;{6};\n'.format(tracktype,t_reset,rz_start,rz_end,landmark,default,stim_amp))
                previous_opto_stim = True

            elif opto_stim == False and previous_opto_stim == True and next_opto_stim_forced == False:
                config_file.write('track;{0};prelick;1.0;gain modulation;1.0;rewarded;1;Reset;{1};RZ start;{2}.0;RZ end;{3}.0;landmark;{4}.0;default;{5};openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n'.format(tracktype,t_reset,rz_start,rz_end,landmark,default))
                previous_opto_stim = False

            elif opto_stim == True and previous_opto_stim == True and next_opto_stim_forced == False:
                config_file.write('track;{0};prelick;1.0;gain modulation;1.0;rewarded;1;Reset;{1};RZ start;{2}.0;RZ end;{3}.0;landmark;{4}.0;default;{5};openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n'.format(tracktype,t_reset,rz_start,rz_end,landmark,default))
                next_opto_stim_forced = True
                previous_opto_stim = False

            elif opto_stim == False and previous_opto_stim == False and next_opto_stim_forced == True:
                config_file.write('track;{0};prelick;1.0;gain modulation;1.0;rewarded;1;Reset;{1};RZ start;{2}.0;RZ end;{3}.0;landmark;{4}.0;default;{5};openloop;-1;bbreset;0;opto_mask;1;opto_stim;{6};\n'.format(tracktype,t_reset,rz_start,rz_end,landmark,default,stim_amp))
                next_opto_stim_forced = False
                previous_opto_stim = True

            elif opto_stim == True and previous_opto_stim == False and next_opto_stim_forced == True:
                config_file.write('track;{0};prelick;1.0;gain modulation;1.0;rewarded;1;Reset;{1};RZ start;{2}.0;RZ end;{3}.0;landmark;{4}.0;default;{5};openloop;-1;bbreset;0;opto_mask;1;opto_stim;{6};\n'.format(tracktype,t_reset,rz_start,rz_end,landmark,default,stim_amp))
                next_opto_stim_forced = True
                previous_opto_stim = True

            elif opto_stim == False and previous_opto_stim == True and next_opto_stim_forced == True:
                config_file.write('track;{0};prelick;1.0;gain modulation;1.0;rewarded;1;Reset;{1};RZ start;{2}.0;RZ end;{3}.0;landmark;{4}.0;default;{5};openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n'.format(tracktype,t_reset,rz_start,rz_end,landmark,default))
                next_opto_stim_forced = True
                previous_opto_stim = False

            elif opto_stim == True and previous_opto_stim == True and next_opto_stim_forced == True:
                config_file.write('track;{0};prelick;1.0;gain modulation;1.0;rewarded;1;Reset;{1};RZ start;{2}.0;RZ end;{3}.0;landmark;{4}.0;default;{5};openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n'.format(tracktype,t_reset,rz_start,rz_end,landmark,default))
                next_opto_stim_forced = True
                previous_opto_stim = False

if __name__ == "__main__":
    mouse = 'LF180919'
    make_config_MTH3_VD('MTH3_s5v2_opto_' + mouse + '_1206', 400, [3,4], 1000, mouse)
    print('done')
