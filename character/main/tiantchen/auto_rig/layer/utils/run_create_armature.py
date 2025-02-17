import os, time

def layer_preprocess(path):
    dirname, _ = os.path.split(os.path.abspath(__file__))
    cmd_1 = " ".join(["/root/tinatchen/blender-3.6.5-linux-x64/blender -b -P {}".format(os.path.join(dirname, "create_armature.py")),"--", path])
    os.system(cmd_1)

if __name__ == '__main__':
    for i in ["yuanmeng"]: 
        "MCWY2_F_T", "MCWY2_M_T", "mcwy_female", "mcwy_male"
        "readyplayerme_male", "readyplayerme_male_T"
        "timer_female", "timer_male"
        "pubg_female", "pubg_male"
        "yuanmeng"
        "quest_female_average", "quest_male_average", "quest_male_fat", "quest_male_slim", "quest_male_strong"
        layer_preprocess(os.path.join("/aigc_cfs/rabbityli/base_bodies", i))