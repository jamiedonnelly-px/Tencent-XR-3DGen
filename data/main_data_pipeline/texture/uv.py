import argparse
import os
import time

import numpy as np
import torch
import xatlas


class Mesh:
    def __init__(self, v_pos=None, t_pos_idx=None, v_nrm=None, t_nrm_idx=None, v_tex=None, t_tex_idx=None, v_tng=None,
                 t_tng_idx=None, material=None, base=None):
        self.v_pos = v_pos  # N, 3 xyz
        self.v_nrm = v_nrm  # N, 3 normal dir
        self.v_tex = v_tex  # N, 2 uv coord
        self.v_tng = v_tng  # N, 3 tangent dir
        self.t_pos_idx = t_pos_idx  # Nf, 3 geom faces, index of v_pos
        self.t_nrm_idx = t_nrm_idx  # Nf, 3 normal faces, index of v_nrm
        self.t_tex_idx = t_tex_idx  # Nf, 3 uv map faces, index of v_tex
        self.t_tng_idx = t_tng_idx  # Nf, 3 tangent map faces, index of v_tng
        self.material = material

        if base is not None:
            self.copy_none(base)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material

    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone().detach()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone().detach()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone().detach()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone().detach()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone().detach()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone().detach()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone().detach()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone().detach()
        return out


def load_obj(filename):
    obj_path = os.path.dirname(filename)

    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load materials
    # all_materials = [
    #     {
    #         'name' : '_default_mat',
    #         'bsdf' : 'pbr',
    #         'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
    #         'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
    #     }
    # ]
    # if not skip_mtl:
    #     if mtl_override is None:
    #         for line in lines:
    #             if len(line.split()) == 0:
    #                 continue
    #             if line.split()[0] == 'mtllib':
    #                 all_materials += material.load_mtl(os.path.join(obj_path, line.split()[1]), clear_ks) # Read in entire material library
    #     else:
    #         all_materials += material.load_mtl(mtl_override)

    # load vertices
    vertices, texcoords, normals = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])

    # load faces
    activeMatIdx = None
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'f':  # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if vv[2] != "" else -1
            for i in range(nv - 2):  # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
    assert len(tfaces) == len(faces) and len(nfaces) == len(faces)

    vertices = torch.tensor(vertices, dtype=torch.float32)
    texcoords = torch.tensor(texcoords, dtype=torch.float32) if len(
        texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32) if len(normals) > 0 else None

    faces = torch.tensor(faces, dtype=torch.int64)
    tfaces = torch.tensor(tfaces, dtype=torch.int64) if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64) if normals is not None else None

    return Mesh(vertices, faces, normals, nfaces, texcoords, tfaces)


######################################################################################
# Save mesh object to objfile
######################################################################################


def write_obj(output_obj_path, mesh):
    print("Writing mesh: ", output_obj_path)
    with open(output_obj_path, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None

        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy(
        ) if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy(
        ) if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy(
        ) if mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))

        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            assert (len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert (len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j] + 1), '' if v_tex is None else str(
                    t_tex_idx[i][j] + 1), '' if v_nrm is None else str(t_nrm_idx[i][j] + 1)))
            f.write("\n")

    print("Done exporting mesh")


def mesh_xatlas(imesh: Mesh):
    device = imesh.v_pos.device
    v_pos = imesh.v_pos.detach().cpu().numpy()  # [N, 3]
    t_pos_idx = imesh.t_pos_idx.detach().cpu().numpy()  # [M, 3]

    # unwrap uvs
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_pos, t_pos_idx)
    chart_options = xatlas.ChartOptions()
    # print('debug chart_options ', chart_options)
    # print_object_attributes(chart_options)
    # chart_options.max_iterations = 0  # 0 disable merge_chart for faster unwrap...
    # chart_options.max_iterations = 10  # 0 disable merge_chart for faster unwrap...
    # chart_options.max_cost = 100
    # chart_options.roundness_weight = 10
    # chart_options.normal_seam_weight = 2000
    # pack_options = xatlas.PackOptions()
    # print('debug pack_options ', pack_options)
    # print_object_attributes(pack_options)
    # pack_options.blockAlign = True
    # pack_options.bruteForce = False
    # atlas.generate(chart_options=chart_options, pack_options=pack_options)
    atlas.generate(chart_options=chart_options)
    vmapping, indices, uvs = atlas[0]  # [N], [M, 3], [N, 2]

    # Convert to tensors, numpy->torch
    # v_pos = v_pos[vmapping]
    indices_int64 = indices.astype(
        np.uint64, casting='same_kind').view(np.int64)
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
    faces = torch.tensor(indices_int64, dtype=torch.int64, device=device)
    new_mesh = Mesh(torch.tensor(v_pos, dtype=torch.float32, device=device).contiguous(),
                    torch.tensor(t_pos_idx, device=device).to(
                        torch.long).contiguous(),
                    v_tex=uvs,
                    t_tex_idx=faces,
                    v_nrm=imesh.v_nrm,
                    t_nrm_idx=imesh.t_nrm_idx,
                    material=None)

    # vmap: remap geom v/f to texture vt/ft, make each v correspond to a unique vt
    vmapping = torch.from_numpy(vmapping.astype(np.int64)).long().to(device)
    # new_mesh.v_pos = new_mesh.v_pos[vmapping]
    # new_mesh.t_pos_idx = new_mesh.t_tex_idx

    return new_mesh


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("UV generation start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='UV generator.')
    parser.add_argument('--source_mesh_path', type=str,
                        help='path of input mesh')
    parser.add_argument('--output_mesh_path', type=str,
                        help='path of output mesh with uv')
    args = parser.parse_args()

    source_mesh_path = args.source_mesh_path
    output_mesh_path = args.output_mesh_path
    # if not os.path.exists(output_mesh_folder):
    #     os.mkdir(output_mesh_folder)

    imesh = load_obj(source_mesh_path)
    new_mesh = mesh_xatlas(imesh)
    write_obj(output_mesh_path, new_mesh)
