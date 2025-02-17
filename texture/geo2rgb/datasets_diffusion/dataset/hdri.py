import cv2
import numpy as np
from scipy.special import sph_harm
import torch

class HDRIcache:
    def __init__(self, hdri_size=(32,64), sh_dim=0):
        self.img_size = hdri_size
        self.hdris = {}
        self.theta = None
        self.phi = None
        self.area = None
        if sh_dim > 0:
            assert int(sh_dim**0.5)**2 == sh_dim and sh_dim**0.5 > 0, "sh_dim must be a positive square number"
            self.sh_degrees = int(sh_dim**0.5) - 1
        else:
            self.sh_degrees = 0
        
    def get(self, path, obj2world):
        obj2world = obj2world.numpy()
        if path not in self.hdris:
            img, self.theta, self.phi, self.area = read_hdri(path, self.img_size)
            self.hdris[path] = img
        img = self.hdris[path]
        theta, phi = rotate_theta_phi(self.theta, self.phi, obj2world)
        img = interpolate_hdri(img, theta, phi)
        if self.sh_degrees > 0:
            # TODO: a much faster way is to precompute unrotated SH coefficients for hdris and rotate them using wigner D matrix
            sh_basis = compute_sh_basis(self.sh_degrees, theta, phi)
            sh_coefficients = project_sh(img, np.expand_dims(sh_basis, -1), self.area)
        else:
            sh_coefficients = np.zeros((0,3), dtype=np.float32)
            
        img = torch.from_numpy(img).float().permute(2,0,1).expand(3,-1,-1) # [3, h, w]
        sh_coefficients = torch.from_numpy(sh_coefficients).float().expand(-1,3)
        
        return img, sh_coefficients
        

def read_hdri(file_path, img_size=None):
    img = cv2.imread(file_path, -1)
    
    if img_size is not None:
        height, width = img_size
        img = cv2.resize(img, (width,height), interpolation=cv2.INTER_AREA)
    
    height, width, *_ = img.shape
    theta = np.linspace(0, np.pi, height, endpoint=False)  # following blender convention: image up is 0, down is pi
    phi = np.mod(np.linspace(3*np.pi, np.pi, width, endpoint=False), 2*np.pi) # following blender convention: from image left-most to right most are: -x, y, x, -y, -x; phi is azimuth with phi=0 being +x and phi=2/pi being +y
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    area = 2 * np.pi / width * np.pi / height * np.sin(theta)
    return img, theta, phi, area

def rotate_theta_phi(theta_object, phi_object, object2world):
    x = np.sin(theta_object) * np.cos(phi_object)
    y = np.sin(theta_object) * np.sin(phi_object)
    z = np.cos(theta_object)
    points = np.stack((x, y, z, np.ones_like(x)), axis=-1)
    world_points = points @ object2world.T
    x_world, y_world, z_world = world_points[..., 0], world_points[..., 1], world_points[..., 2]
    r_world = np.sqrt(x_world**2 + y_world**2 + z_world**2)
    theta_world = np.arccos(z_world / r_world)
    phi_world = np.arctan2(y_world, x_world)
    phi_world = np.mod(phi_world, 2*np.pi)  
    return theta_world, phi_world

def interpolate_hdri(img, theta, phi):
    height, width, *_ = img.shape
    y = theta / np.pi * height
    x = (1 - phi / (2*np.pi)) * width
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
    

def compute_sh_basis(degree, theta, phi):
    
    ls = []
    ms = []
    
    for l in range(degree+1):
        for m in range(-l, l+1):
            ls.append(l)
            ms.append(m)
            
    l = np.array(ls)
    m = np.array(ms)
    
    Yml = sph_harm(np.expand_dims(abs(m), (1,2)), np.expand_dims(l, (1,2)), phi, theta)
    
    ret = np.zeros((l.shape[0],)+theta.shape)
    ret[m<0] = (np.sqrt(2) * (-1.0)**np.expand_dims(m, (1,2)) * Yml.imag)[m<0]
    ret[m>0] = (np.sqrt(2) * (-1.0)**np.expand_dims(m, (1,2)) * Yml.real)[m>0]
    ret[m==0] = Yml.real[m==0]
    
    return ret
    

def project_sh(img, basis, area):
    return np.sum(img * np.expand_dims(area, axis=-1) * basis, axis=(1,2))

def unproject_sh(coefficients, basis):
    return np.sum(basis * np.expand_dims(coefficients, axis=(1,2)), axis=0)

