import json


base_body_map = {
    "male": {
        "mcwy2": {
            "path": "/aigc_cfs/rabbityli/base_bodies/MCWY2_M_T",
            "profile": "/aigc_cfs/rabbityli/base_bodies/MCWY2_M_T/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "修长"
        },
        "mcwy1": {
            "path": "/aigc_cfs/rabbityli/base_bodies/mcwy_male",
            "profile": "/aigc_cfs/rabbityli/base_bodies/mcwy_male/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption": "匀称"
        },
        "pubg": {
            "path": "/aigc_cfs/rabbityli/base_bodies/pubg_male",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "成年人"
        },
        "timer": {
            "path": "/aigc_cfs/rabbityli/base_bodies/timer_male",
            "profile": "/aigc_cfs/rabbityli/base_bodies/timer_male/profile.png",
            "shoes_scale": 3.0,
            "use_shoes": False,
            "use_hair": True,
            "body_caption":
                "小孩儿"
        },
        "quest_average": {
            "path": "/aigc_cfs/rabbityli/base_bodies/quest_male_average",
            # "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "普通"
        },
        "quest_slim": {
            "path": "/aigc_cfs/rabbityli/base_bodies/quest_male_slim",
            # "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "瘦弱"
        },
        "quest_strong": {
            "path": "/aigc_cfs/rabbityli/base_bodies/quest_male_strong",
            # "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "强壮"
        },
        "quest_fat": {
            "path": "/aigc_cfs/rabbityli/base_bodies/quest_male_fat",
            # "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "肥胖"
        },

    },
    "female": {
        "mcwy2": {
            "path": "/aigc_cfs/rabbityli/base_bodies/MCWY2_F_T",
            "profile": "/aigc_cfs/rabbityli/base_bodies/MCWY2_F_T/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "修长"
        },
        "mcwy1": {
            "path": "/aigc_cfs/rabbityli/base_bodies/mcwy_female",
            "profile": "/aigc_cfs/rabbityli/base_bodies/mcwy_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "匀称"
        },
        "pubg": {
            "path": "/aigc_cfs/rabbityli/base_bodies/pubg_female",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "成年人"
        },
        "timer": {
            "path": "/aigc_cfs/rabbityli/base_bodies/timer_female",
            "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 3.0,
            "use_shoes": False,
            "use_hair": True,
            "body_caption":
                "小孩儿"

        },
        "quest_average": {
            "path": "/aigc_cfs/rabbityli/base_bodies/quest_female_average",
            # "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "普通"
        },
        "quest_slim": {
            "path": "/aigc_cfs/rabbityli/base_bodies/quest_male_slim",
            # "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "瘦弱"
        },
        "quest_strong": {
            "path": "/aigc_cfs/rabbityli/base_bodies/quest_male_strong",
            # "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "强壮"
        },
        "quest_fat": {
            "path": "/aigc_cfs/rabbityli/base_bodies/quest_male_fat",
            # "profile": "/aigc_cfs/rabbityli/base_bodies/timer_female/profile.png",
            "shoes_scale": 1.0,
            "use_shoes": True,
            "use_hair": True,
            "body_caption":
                "肥胖身材"
        },
    }
}

yuanmeng = {
    "path": "/aigc_cfs/rabbityli/base_bodies/yuanmeng",
    "shoes_scale": 3.0,
    "use_shoes": False,
    "use_hair": False,
    "body_caption":
        "圆滚滚, chubby, round, ball, goat, sheep"
}



base_body_map["male"]["yuanmeng"] = yuanmeng
base_body_map["female"]["yuanmeng"] = yuanmeng


def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data

def write_json(fname,j):
    json_object = json.dumps(j, indent=4)
    with open( fname, "w") as outfile:
        outfile.write(json_object)



if __name__ == '__main__':


    write_json( "./base_body_map.json" , base_body_map )

