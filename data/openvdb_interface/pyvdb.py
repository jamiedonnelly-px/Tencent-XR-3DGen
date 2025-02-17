import pyopenvdb as vdb
import trimesh
import numpy as np
from pdb import set_trace as st
import os

def export_mesh_to_obj(points, triangles, quads, filename):
    with open(filename, 'w') as file:
        for point in points:
            file.write(f"v {point[0]} {point[1]} {point[2]}\n")

        for tri in triangles:
            file.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

        for quad in quads:
            file.write(f"f {quad[0] + 1} {quad[1] + 1} {quad[2] + 1} {quad[3] + 1}\n")

def normalize_to_bounding_box(points):
    
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)

    center = (min_vals + max_vals) / 2.0
    scale = (max_vals - min_vals).max() / 2.0

    normalized_points = (points - center) / scale

    return normalized_points

class VDBGrid:
    
    def __init__(self, grid):
        
        self.grid = grid
        
        self.if_centre = np.asarray(self.grid.transform.worldToIndex((0,0,0)))
        self.voxel_size = np.asarray(self.grid.transform.voxelSize())
        
    def _check_transform(self):
        
        assert self.grid.transform.isLinear
        
        for ifc in zip([
            (1,0,0),
            (0,1,0),
            (0,0,1),
        ]):
            wdc = self.transform.indexToWorld(ifc)
            assert np.all_close(ifc, self.transform.worldToIndex(wdc))
            assert np.all_close((-self.if_centre + ifc) * voxel_size, wdc)
    
    @staticmethod        
    def load_vdb(filepath):
        '''
        read vdb file from disk
        
        inputs:
            - filepath: path to vdb file
        
        returns:
            - a VDBGrid object
        '''
        return VDBGrid(vdb.read(filepath))
    
    @staticmethod
    def load_mesh(filepath, normalize=True, voxelsize=1/256, bandwidth=3):
        '''
        read watertight mesh from disk and convert to vdb
        
        inputs:
            - filepath: path to mesh file
            - normalize: whether or not to normalize mesh to a bounding box of [-1,1]^3 before voxelization
            - voxelsize: vdb voxel size
            - bandwidth: defines a distance margin from isosurface where voxels within this band are 'active'.
                         VDB grids only record active voxels' SDF while out-of-band voxels are inactive and their value truncated. 
                         e.g. bandwidth=3 means 3 voxels from both inside and outside mesh surface are active.
        
        returns:
            - a VDBGrid object if mesh is watertight and non-empty; otherwise returns None
        '''
        transform = vdb.createLinearTransform(voxelSize=voxelsize)
        mesh = trimesh.load(filepath, process=False)
        verts = mesh.vertices
        faces = mesh.faces
        
        # make unique
        verts, vmap = np.unique(verts, return_inverse=True, axis=0)
        faces = vmap[faces]
        if normalize:
            verts = normalize_to_bounding_box(verts)
            
        water_tight = trimesh.Trimesh(verts, faces, process=False).is_watertight
        
        if not water_tight:
            return None
        
        grid = vdb.FloatGrid.createLevelSetFromPolygons(verts, faces, transform=transform, halfWidth=bandwidth)
        
        is_empty = True
        for iter in grid.iterOnValues():
            is_empty = False
            break
        
        if is_empty:
            return None
        
        return VDBGrid(grid)
    
    def export_vdb(self, filepath):
        '''
        export vdb to disk
        '''
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        vdb.write(filepath, grid=grid)
    
    def export_mesh(self, filepath, isovalue=0.0, adaptivity=0.0, flip=True):
        '''
        extract isosurface from vdb and export a watertight obj
        
        inputs:
            - filepath: export path to mesh
            - isovalue: sdf values for isosurface extraction in world unit.
                        by default this should be 0; positive values result in expanded mesh 
                        and negative values result in eroded mesh
            - adaptivity: value between [0,1]; 0 results in pure quad mesh from dual contouring,
                        greater values produce simplified mesh with less faces that may be triangle or quad
            - flip: whether to flip the orientation of mesh
        '''
        
        points, triangles, quads = grid.convertToPolygons(isovalue=isovalue, adaptivity=adaptivity)
        
        if flip:
            triangles = [np.asarray(tri)[::-1].tolist() for tri in triangles]
            quads = [np.asarray(quad)[::-1].tolist() for quad in quads]
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        export_mesh_to_obj(points, triangles, quads, filepath)
    
    def sample_coarse2fine(self, d):
        '''
        perform coarse-to-fine sampling on a VDB grid by first downsampling active regions at a lower resolution
        and then refining each active coarse voxel into a dense regular subgrid of smaller voxels. 
        
        returns the coordinates of the coarse voxels and the corresponding values for the subgrids.
        
        inputs:
            - d: downsampling factor of coarse voxel, int
        
        returns a tuple of:
            - coarse_voxel_xyz: N x 3 where N is number of sparse coarse voxels that are at least partially active
            - subdivision_offset: d x d x d x 3, coordinate offsets of subdivision to coarse voxel xyz. such that
                coarse_voxel_xyz.reshape(N,1,1,1,3) + subdivision_offset.reshape(1,d,d,3) gives the world coordinates
                of fine level voxels
            - sdf: N x d x d x d, SDF values of fine voxels. note this can contain inactive SDF values that are truncated
                to bandWith
            - sdf_grad: N x d x d x d x 3, spatial gradients of truncated sdf above. values are NOT normalized.
        '''

        active_is_coords = self.grid.activeIndices().astype(np.int32) 
        coarse_is_coords = np.unique(active_is_coords // d * d, axis=0) # [N,3]
        
        fine_is_offset = np.stack(np.meshgrid(
                np.arange(d),
                np.arange(d),
                np.arange(d),
                indexing="ij",
            ), axis=-1).astype(coarse_is_coords.dtype) # [d,d,d,3]
        
        fine_is_coords = coarse_is_coords.reshape(-1,1,1,1,3) + fine_is_offset
        
        fine_sdf, fine_sdf_grad = self.grid.queryValGradInt(fine_is_coords.reshape(-1,3))
        
        fine_if_offset = fine_is_offset - (d - 1) / 2 # [d,d,d,3]
        coarse_if_coords = coarse_is_coords.astype(np.float32) + (d - 1) / 2 # [N,3]
        fine_wd_offset = fine_if_offset * self.voxel_size
        coarse_wd_coords = (coarse_if_coords - self.if_centre) * self.voxel_size
        
        return coarse_wd_coords, fine_wd_offset, fine_sdf.reshape(-1,d,d,d), fine_sdf_grad.reshape(-1,d,d,d,3)


def export_sampling(points, sdf, vectors, filename):
    """
    Export 3D points, SDF values, and vectors to a PLY file for MeshLab visualization.

    Args:
        points (np.ndarray): Array of 3D points of shape [N, 3].
        sdf (np.ndarray): SDF values of shape [N].
        vectors (np.ndarray): Vector data of shape [N, 3].
        filename (str): Output PLY file path.
    """
    assert points.shape[1] == 3, "Points array must have shape [N, 3]"
    assert sdf.shape[0] == points.shape[0], "SDF array must have the same length as points"
    assert vectors.shape == points.shape, "Vectors array must have shape [N, 3]"
    
    # Create RGB colors based on SDF sign
    colors = np.zeros((sdf.shape[0], 3), dtype=np.uint8)
    colors[sdf < 0] = [0, 0, 255]  # Blue for negative SDF
    colors[sdf >= 0] = [255, 0, 0]  # Red for nonnegative SDF
    
    # Write to PLY file
    with open(filename, 'w') as file:
        # PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("property float nx\n")
        file.write("property float ny\n")
        file.write("property float nz\n")
        file.write("end_header\n")
        
        # Write vertex data
        for i in range(points.shape[0]):
            file.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} ")
            file.write(f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} ")
            file.write(f"{vectors[i, 0]} {vectors[i, 1]} {vectors[i, 2]}\n")


if __name__ == "__main__":
    
    
    vdb = VDBGrid.load_mesh("example/statue.obj", voxelsize=1/256, bandwidth=3)
    vdb.export_mesh("example/statue_dc.obj")
    coarse_xyz, fine_offset, sdf, sdf_grad = vdb.sample_coarse2fine(8)
    export_sampling((coarse_xyz.reshape(-1,1,1,1,3)+fine_offset).reshape(-1,3), sdf.flatten(), sdf_grad.reshape(-1,3), "example/statue.ply")

    
    
    