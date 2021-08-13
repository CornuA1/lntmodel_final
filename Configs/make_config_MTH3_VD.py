"""
Create a config file with given parameters. Primarily used to get a random sequence of tracks

"""

import random

def make_config_MTH3_VD(fname, lines, tracknrs, mousename=[]):
    if not mousename:
        mousename = 'VD_mouse'

    with open("{0}.csv".format(fname), 'w') as config_file:
        config_file.write('Experiment;MTH3_VD;Length (min);30;Day;1;ExpGroup;1;Mouse;{0};DOB;00000000;Strain;C57WT;Stop Threshold;0.7;Valve Open Time;0.3;Comments;no\n'.format(mousename))
        for i in range(lines):
            track = random.choice(tracknrs)
            if track == 7:
                config_file.write("track;7;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;205.0;RZ end;245.0;landmark;145.0;default;245;openloop;-1;bbreset;0;\n")
            else:
                config_file.write("track;9;prelick;1.0;gain modulation;1.0;rewarded;0;Reset;400;RZ start;205.0;RZ end;245.0;landmark;145.0;default;245;openloop;-1;bbreset;0;\n")

            config_file.write("track;{0};prelick;1.0;gain modulation;1.0;rewarded;{1};Reset;400;RZ start;205.0;RZ end;245.0;landmark;205.0;default;245;openloop;-1;bbreset;0;\n".format(thistrack, rew))

if __name__ == "__main__":
    make_config_MTH3_VD('MTH3_VD', 100, [7,9], 'VD_mouse')
