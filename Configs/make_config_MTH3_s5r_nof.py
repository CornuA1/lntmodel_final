"""
Create a config file with given parameters. Primarily used to get a random sequence of tracks

"""

import random

def make_config_MTH3_VD(fname, lines, tracknrs, mousename=[]):
    if not mousename:
        mousename = 'C57'

    with open("{0}.csv".format(fname), 'w') as config_file:
        config_file.write('Experiment;MTH3_vr1_s5r2;Length (min);30;Day;1;ExpGroup;1;Mouse;{0};DOB;00000000;Strain;C57;Stop Threshold;0.7;Valve Open Time;0.3;Comments;no\n'.format(mousename))
        for i in range(lines):
            track = random.choice(tracknrs)
            if track == 3:
                if random.choice([1,2,3,4,5]) > 3:
                    config_file.write("track;11;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;\n")
                else:
                    config_file.write("track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;340.0;landmark;240.0;default;340;openloop;-1;bbreset;0;\n")
            else:
                if random.choice([1,2,3,4,5]) > 3:
                    config_file.write("track;12;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;460;RZ start;380.0;RZ end;400.0;landmark;240.0;default;400;openloop;-1;bbreset;0;\n")
                else:
                    config_file.write("track;4;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;460;RZ start;380.0;RZ end;400.0;landmark;240.0;default;400;openloop;-1;bbreset;0;\n")

            #config_file.write("track;{0};prelick;1.0;gain modulation;1.0;rewarded;{1};Reset;400;RZ start;205.0;RZ end;245.0;landmark;205.0;default;245;openloop;-1;bbreset;0;\n".format(thistrack, rew))

if __name__ == "__main__":
    make_config_MTH3_VD('MTH3_s5r_nof', 400, [3,4], 'C57')
