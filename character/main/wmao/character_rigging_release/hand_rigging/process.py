import numpy as np
import os
from ipdb import set_trace as st
import trimesh
import open3d as o3d
from matplotlib import cm

def export_color_obj(mesh, out_file, vertex_colors):
    # Export the mesh to OBJ
    obj_data = trimesh.exchange.obj.export_obj(mesh)

    # Modify the OBJ file to include vertex colors
    lines = obj_data.splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("v "):  # Vertex line
            # Append vertex color information
            idx = len(new_lines)
            color = vertex_colors[idx % len(vertex_colors)]  # Prevent overflow
            line += f" {color[0]} {color[1]} {color[2]} {color[3]}"
        new_lines.append(line)

    # Save the modified OBJ
    with open(out_file, "w") as f:
        f.write("\n".join(new_lines))

def get_hand_part(verts, weights, joint_names):
    part_colors = np.array([[1,1,1,1], [1,0,0,1], [0,1,0,1], [0,0,1,1], [1, 1, 0,1], [1, 0, 1,1]]) * 1.0
    color = np.zeros([verts.shape[0], 4])
    joint2parts = []
    for ji, jn in enumerate(joint_names):
        if jn.endswith('Hand'):
            pid = 0
        elif 'Thumb' in jn:
            pid = 1
        elif 'Index' in jn:
            pid = 2
        elif 'Middle' in jn:
            pid = 3
        elif 'Ring' in jn:
            pid = 4
        elif 'Pinky' in jn:
            pid = 5
        else:
            assert False
        joint2parts.append(pid)

    joint2parts = np.array(joint2parts)
    wmax_jid = np.argmax(weights, axis=1)
    vert_parts = joint2parts[wmax_jid]
    color = part_colors[vert_parts]
    return color, vert_parts
    # vids = np.where(wmax_jid==lji)[0]
    # left_color[vids] = part_colors[pid]
    # out_file = os.path.join(out_dir, 'debug', file_name.split('.npz')[0]+ '_' + joint_names[ji] +'.ply')
    # draw_weights(weights_left_hand[:,lji],verts_left_hand,faces_left_hand,out_file)

def draw_weights(weights, verts, faces, out_file='./'):
    colormap = cm.jet
    col = colormap(weights)[:,:3]
    print(col.shape, col.min(), col.max())
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(col)
    o3d.io.write_triangle_mesh(out_file, mesh)

data_path = '/aigc_cfs_2/weimao/hand_rigging/mixamo_character/npzs'
out_dir = '/aigc_cfs_2/weimao/hand_rigging/data/npz/'
out_dir_vis = '/aigc_cfs_2/weimao/hand_rigging/data/vis/'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir_vis, exist_ok=True)

for file_name in os.listdir(data_path):
    # st()
    if not file_name.endswith('.npz'):
        continue
    data = np.load(os.path.join(data_path,file_name),allow_pickle=True)

    verts = data['verts']
    faces = data['faces']
    weights = data['weights']
    joint_names = data['joint_names']
    parents = data['parents']
    joints = data['joints']
    part_vert_num = data['part_vert_num']

    left_joint_ids = []
    right_joint_ids = []
    for ji, jn in enumerate(joint_names):
        if 'LeftHand' in jn:
            left_joint_ids.append(ji)
        if 'RightHand' in jn:
            right_joint_ids.append(ji)

    left_hand_vids = np.where(weights[:,left_joint_ids].sum(axis=1)>=0.5)[0]
    right_hand_vids = np.where(weights[:,right_joint_ids].sum(axis=1)>=0.5)[0]
    if len(left_hand_vids) > 50:

        vid2leftvid = np.zeros(verts.shape[0]) - 1
        vid2leftvid[left_hand_vids] = np.arange(len(left_hand_vids)) 
        verts_left_hand = verts[left_hand_vids]
        faces_left_hand = vid2leftvid[faces]
        faces_left_hand = faces_left_hand[np.all(faces_left_hand>=0, axis=1)]

        vert_colors, vert_parts = get_hand_part(verts_left_hand, weights[left_hand_vids][:,left_joint_ids], joint_names[left_joint_ids])
        
        out_file = os.path.join(out_dir, file_name.split('.npz')[0]+'_left.npz')
        np.savez_compressed(out_file, 
                            verts=verts_left_hand, 
                            faces=faces_left_hand, 
                            weights=weights[left_hand_vids][:,left_joint_ids], 
                            joints=joints[left_joint_ids], 
                            joint_names=joint_names[left_joint_ids], 
                            vert_parts=vert_parts)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_left_hand)
        mesh.triangles = o3d.utility.Vector3iVector(faces_left_hand)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors[:,:3])
        out_file = os.path.join(out_dir_vis, file_name.split('.npz')[0]+'_left.ply')
        o3d.io.write_triangle_mesh(out_file, mesh)

        # out_file = os.path.join(out_dir, 'debug', file_name.split('.npz')[0]+ '_' + joint_names[ji] +'.ply')
        # draw_weights(weights_left_hand[:,lji],verts_left_hand,faces_left_hand,out_file)

        # mesh = trimesh.Trimesh(vertices=verts_left_hand, faces=faces_left_hand)#,vertex_colors=np.uint8(left_color*255))
        # out_file = os.path.join(out_dir, file_name.split('.npz')[0]+'_left.obj')
        # export_color_obj(mesh, out_file, vert_colors)



    if len(right_hand_vids) > 50:
        vid2rightvid = np.zeros(verts.shape[0]) - 1
        vid2rightvid[right_hand_vids] = np.arange(len(right_hand_vids)) 
        verts_right_hand = verts[right_hand_vids]
        faces_right_hand = vid2rightvid[faces]
        faces_right_hand = faces_right_hand[np.all(faces_right_hand>=0, axis=1)]


        vert_colors, vert_parts = get_hand_part(verts_right_hand, weights[right_hand_vids][:,right_joint_ids], joint_names[right_joint_ids])

        out_file = os.path.join(out_dir, file_name.split('.npz')[0]+'_right.npz')
        np.savez_compressed(out_file, 
                            verts=verts_right_hand, 
                            faces=faces_right_hand, 
                            weights=weights[right_hand_vids][:,right_joint_ids], 
                            joints=joints[right_joint_ids], 
                            joint_names=joint_names[right_joint_ids], 
                            vert_parts=vert_parts)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_right_hand)
        mesh.triangles = o3d.utility.Vector3iVector(faces_right_hand)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors[:,:3])
        out_file = os.path.join(out_dir_vis, file_name.split('.npz')[0]+'_right.ply')
        o3d.io.write_triangle_mesh(out_file, mesh)
