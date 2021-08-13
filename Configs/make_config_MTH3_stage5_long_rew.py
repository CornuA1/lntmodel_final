"""
Create a config file with given parameters. Primarily used to get a random sequence of tracks

"""

import random

def make_config_MTH3_VD(fname, lines, tracknrs, mousename=[]):
    if not mousename:
        mousename = 'C57'

    with open("{0}.csv".format(fname), 'w') as config_file:
        config_file.write('Experiment;MTH3_vr1_s5lr;Length (min);60;Day;1;ExpGroup;1;Mouse;{0};DOB;00000000;Strain;C57;Stop Threshold;0.7;Valve Open Time;0.1;Comments;no\n'.format(mousename))
        for i in range(lines):
            track = random.choice(tracknrs)
            if track == 3:
                config_file.write("track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;410;RZ start;320.0;RZ end;400.0;landmark;240.0;default;400;openloop;-1;bbreset;0;\n")
            else:
                config_file.write("track;4;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;490;RZ start;380.0;RZ end;480.0;landmark;240.0;default;480;openloop;-1;bbreset;0;\n")

            #config_file.write("track;{0};prelick;1.0;gain modulation;1.0;rewarded;{1};Reset;400;RZ start;205.0;RZ end;245.0;landmark;205.0;default;245;openloop;-1;bbreset;0;\n".format(thistrack, rew))

if __name__ == "__main__":
    make_config_MTH3_VD('MTH3_s5lr', 300, [3,4], 'C57')
