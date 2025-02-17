import numpy as np

neck_border = [1949, 1326, 553, 552, 2484, 220, 737, 497, 1727, 1726, 1931, 3013, 1932, 2219, 2036, 496, 649, 563, 13, 2149, 1940]

all_dof = list(np.arange(63))
mutelist = {
    "smpl+d" :   [34, 35] +  # Neck yz-axis
                 [43, 44] +  # Head yz-axis
                 [7, 8] +  # Spin1 yz-axis
                 [16, 17] +  # Spin2 yz-axis
                 [25, 26] +  # Spin3 yz-axis
                 [18, 19, 20] +  # footl all-axis
                 [21, 22, 23] +  # footR all-axis
                 [27, 28, 29, 30, 31, 32] + # toes l&r all axis
                 [57, 58, 59, 60, 61, 62] ,# all hand

    "vroid":  [ x for x in all_dof if x not in [ 0, 1,  3, 4 ] ],  # Tpose, only free thigh

    "daz":  [ x for x in all_dof if x not in [ 45, 46, 48, 49  ]], # , 45, 46, 48, 49] ],  # A pose  free thigh, and shoulder

}


pose_init = {
    "daz" : {
        47: -0.8,
        50: 0.8,
        2: 0.1,
        5: -0.1
    }

}



SMPLX_JOINT_NAMES = [
            'thigh_l', 'thigh_r', # 0 1, 2, 3,4 5
            'spine1', # 6 7 8
            'calf_l','calf_r', # 9 10 11 12 13 14
            'spine2', #  15 16 17
            'foot_l','foot_r', # 18 19 20 21 22 23
            'spine3', # 24 25 26
            'toes_l','toes_r', # 27 28 29 30 31 32
            'neck', # 33 34 35
            'clavicle_l','clavicle_r', # 36 37 38 9 0 1
            'head', # 42 3 4
            'upperarm_l','upperarm_r',  # 45 6 7 8 9 50
            'lowerarm_l', 'lowerarm_r', # 51 2 3 4 5 6
            'hand_l','hand_r' # 57 8 9 60 1 2
            ]
