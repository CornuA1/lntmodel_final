"""
Create a config file with given parameters. Primarily used to get a random sequence of tracks

"""

import random

def make_config_MTH3_VD(fname, lines, tracknrs, stim_amp, mousename=[]):
    if not mousename:
        mousename = 'C57'

    with open("{0}.csv".format(fname), 'w') as config_file:
        config_file.write('Experiment;MTH3_vr2_opto;Length (min);60;Day;5;ExpGroup;1;Mouse;{0};DOB;00000000;Strain;C57;Stop Threshold;0.7;Valve Open Time;0.3;Comments;opto\n'.format(mousename))
        previous_opto_stim = False
        next_opto_stim_forced = False
        for i in range(lines):
            track = tracknrs[0]
            # 1:2 chance to have the masking stimulus on
            if random.choice([False,True]):
                opto_stim = random.choice([False,True])
                if opto_stim == False and previous_opto_stim == False and next_opto_stim_forced == False:
                    config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n')

                elif opto_stim == True and previous_opto_stim == False and next_opto_stim_forced == False:
                    config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;1;opto_stim;{0};\n'.format(stim_amp))
                    previous_opto_stim = True

                elif opto_stim == False and previous_opto_stim == True and next_opto_stim_forced == False:
                    config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n')
                    previous_opto_stim = False

                elif opto_stim == True and previous_opto_stim == True and next_opto_stim_forced == False:
                    config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n')
                    next_opto_stim_forced = True
                    previous_opto_stim = False

                elif opto_stim == False and previous_opto_stim == False and next_opto_stim_forced == True:
                    config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;1;opto_stim;{0};\n'.format(stim_amp))
                    next_opto_stim_forced = False
                    previous_opto_stim = True

                elif opto_stim == True and previous_opto_stim == False and next_opto_stim_forced == True:
                    config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;1;opto_stim;{0};\n'.format(stim_amp))
                    next_opto_stim_forced = True
                    previous_opto_stim = True

                elif opto_stim == False and previous_opto_stim == True and next_opto_stim_forced == True:
                    config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n')
                    next_opto_stim_forced = True
                    previous_opto_stim = False

                elif opto_stim == True and previous_opto_stim == True and next_opto_stim_forced == True:
                    config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;1;opto_stim;0;\n')
                    next_opto_stim_forced = True
                    previous_opto_stim = False


            else:
                config_file.write('track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;opto_mask;0;opto_stim;0;\n')
                previous_opto_stim = False

if __name__ == "__main__":
    make_config_MTH3_VD('MTH3_s2_opto_4', 400, [3], 2048, 'C57')
