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
                if random.choice([1,2,3]) == 3 and i > 20:
                    config_file.write("track;13;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;500;RZ start;420.0;RZ end;480.0;landmark;340.0;default;480;openloop;-1;bbreset;0;\n")
                else:
                    config_file.write("track;3;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;400;RZ start;320.0;RZ end;380.0;landmark;240.0;default;380;openloop;-1;bbreset;0;\n")
            else:
                if random.choice([1,2,3]) == 3 and i > 20:
                    config_file.write("track;12;prelick;1.0;gain modulation;0.5;rewarded;1;Reset;600;RZ start;480.0;RZ end;540.0;landmark;340.0;default;540;openloop;-1;bbreset;0;\n")
                else:
                    config_file.write("track;4;prelick;1.0;gain modulation;1.0;rewarded;1;Reset;500;RZ start;380.0;RZ end;440.0;landmark;240.0;default;440;openloop;-1;bbreset;0;\n")

            #config_file.write("track;{0};prelick;1.0;gain modulation;1.0;rewarded;{1};Reset;400;RZ start;205.0;RZ end;245.0;landmark;205.0;default;245;openloop;-1;bbreset;0;\n".format(thistrack, rew))

if __name__ == "__main__":
    make_config_MTH3_VD('MTH3_s5r_dl', 400, [3,4], 'C57')
