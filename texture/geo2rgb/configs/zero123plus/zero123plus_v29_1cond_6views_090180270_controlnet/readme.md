# optimize:
    self attention: yes
    cross attention kv weight: yes
    ramping_coefficients: yes

# pretrain model:
    origin stable diffusion 2 checkpoint as pretrain model

# target type:
    shading

# data:
    target 4 view: 
        elevation[0, 0, 0, 0],
        azimuth[0, 90, 180, 270]
    condition:
        azimuth 0-360
        elevation: random from [-15, 30]

# info:
    0,90,180,270 pose of target images
    0.1 percentage for cfg