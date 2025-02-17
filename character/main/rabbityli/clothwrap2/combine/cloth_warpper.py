import copy
import os, sys

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)

from lib import *

base_body_map = load_json(os.path.join(codedir, "body_config/base_body_map.json"))
base_body_map_source = load_json(os.path.join(codedir,"body_config/base_body_map_source.json"))
key_smpl_map = load_json(os.path.join(codedir, "body_config/20240711_key_smpl_map.json"))

ruku_ok = load_json(os.path.join(codedir,  "part/20240925_gdp.json") )["data"]
clses = ruku_ok.keys()
ruku_map = {}
for cls in clses:
    ruku_map.update( ruku_ok[cls] )


G_trns = np.eye(4)
G_trns[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()


def colorize( warp_visual_lst, pensol_visual_lst, body_manifold  ):

    body_manifold.paint_uniform_color([0.7, 0.7, 0.7])

    warp_visual_body = copy.deepcopy( body_manifold )
    pensol_visual_body = copy.deepcopy( body_manifold )

    for i in range( len(warp_visual_lst)):

        warp_visual, pensol_visual = warp_visual_lst[i], pensol_visual_lst[i]

        warp_visual = smplify_mesh(warp_visual[0], warp_visual[1], clst[i])
        pensol_visual = smplify_mesh(pensol_visual[0], pensol_visual[1], clst[i])

        warp_visual_body = warp_visual + warp_visual_body
        pensol_visual_body = pensol_visual + pensol_visual_body


    return warp_visual_body, pensol_visual_body


def merge_assets( warp_visual_lst,  asset_lst ):
    meshes = [ ]
    for i in range ( len(warp_visual_lst) ):
        m = copy.deepcopy( asset_lst[i].visual_mesh)
        m.vertices = warp_visual_lst[i][0]
        meshes.append(m)

    m = trimesh.util.concatenate(meshes)
    return m






def run_warp(asset_body_path_lst, asset_proxy_path_lst, asset_visual_path_lst, asset_label_lst, target_body_path, lst_name, dump_path_lst, device):


    T_body = Body(target_body_path,  device=device)

    pensol_visual_lst = []
    warp_visual_lst = []
    asset_lst = []

    parts = {
        "top": None,
        "trousers": None,
        "l-shoe": None,
        "r-shoe": None,
        "outfit": None,
        "others": None,
        "hair": None
    }

    for i in range ( len( asset_visual_path_lst)):

        asset_body_path, asset_proxy_path, asset_visual_path, asset_label = \
            asset_body_path_lst[i], asset_proxy_path_lst[i], asset_visual_path_lst[i], asset_label_lst[i],


        S_body = Body(asset_body_path, device=device)
        T_s2t = T_body.T @ torch.inverse(S_body.T)
        asset = Asset(asset_proxy_path, asset_visual_path, G_trns, label=asset_label, device = device)

        ###########################################################################
        ############## Wrap cloth and preserve laplacian ##########################
        ###########################################################################
        offsets, proxy_vert_wrap, warp_visual_vert = warp_asset_with_proxies(asset, S_body, T_body, T_s2t)
        warp_visual = [warp_visual_vert, asset.visual_faces]


        ###########################################################################
        ############## Solve cloth body penetration ###############################
        ###########################################################################
        pensol_visual, pensol_proxy, pensol_proxy_lst = solve_asset_body_penetration(asset, proxy_vert_wrap, T_body, warp_visual_vert)

        warp_visual_lst.append(warp_visual)
        pensol_visual_lst.append(pensol_visual)


        # collect transformed asset
        asset.data["Vs"] = pensol_proxy_lst
        asset.visual_verts = pensol_visual[0]
        parts[asset_label] = [asset, i]


        asset_lst.append(asset)


    # warp_visual_body, pensol_visual_body = colorize( warp_visual_lst, pensol_visual_lst, T_body.body_manifold )
    # o3d.io.write_triangle_mesh(os.path.join( "output/",  "warp_" + lst_name + ".ply"), warp_visual_body)
    # o3d.io.write_triangle_mesh(os.path.join( "output/",  "pensol_visual" + lst_name + ".ply"), pensol_visual_body)
    # o3d.io.write_triangle_mesh(os.path.join( "output/",  "body.ply"), T_body.body_manifold)
    # m = merge_assets(pensol_visual_lst, asset_lst)
    # m.export( os.path.join( "output/",  "pensol_visual" + lst_name + ".obj") )


    ###########################################################################
    ############## Solve cloth-cloth penetration ##############################
    ###########################################################################
    collision_pairs = [
        ["outfit", "trousers", "above"],
        ["r-shoe", "trousers", "adaptive"],
        ["l-shoe", "trousers", "adaptive"]
    ]
    for pair in collision_pairs:

        if parts[pair[0]] and parts[pair[1]]:

            S_asset, sid = parts[pair[0]]
            T_asset, tid = parts[pair[1]]

            mode = pair[2]

            pensol_visual, pensol_proxy, pensol_proxy_lst = solve_asset_asset_penetration( S_asset, T_asset, T_body, Mode = mode)

            S_asset.data["Vs"] = pensol_proxy_lst
            S_asset.visual_verts = pensol_visual[0]
            pensol_visual_lst[sid] = pensol_visual



    ### export assets
    for idx, ast in enumerate ( asset_lst ):
        ast.update_mesh( )
        ast.export_mesh(dump_path_lst[idx])


    # warp_visual_body, pensol_visual_body = colorize( warp_visual_lst, pensol_visual_lst, T_body.body_manifold )
    # # # o3d.io.write_triangle_mesh(os.path.join( "output/",  "p2p_pensol_visual" + lst_name + ".ply"), pensol_visual_body)
    # o3d.io.write_triangle_mesh(os.path.join("output/", lst_name + ".ply"), pensol_visual_body)



def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--lst_path", type=str, required=True)
    args = parser.parse_args()

    lst_path = args.lst_path

    lst_name = lst_path.split("/")[-1][:-5]

    dump_root = Path(lst_path).parent

    device = torch.device(0)

    with open(lst_path, "rb") as f:
        data = json.load(f)
        part_info = data["path"]
        body_info = data["body_attr"]

        print("body_info", body_info)

    if len(part_info) == 0:
        print("no items in the list")
        exit()


    target_dir = base_body_map [body_info[0]][body_info[1]]["path"]
    target_body_path = os.path.join(target_dir, "smplx_and_offset_smplified.npz")


    asset_body_path_lst = []
    asset_proxy_path_lst = []
    asset_visual_path_lst = []
    asset_label_lst = []
    dump_path_lst = []
    warp_lst = {}

    for idx, part in enumerate(part_info):

        asset_key = part_info[part]["asset_key"]
        asset_info = ruku_map[asset_key ]
        label = asset_info["Category"]  # categories

        if label == "others":
            continue

        ### check hair and shoes:
        if label == "shoe" and base_body_map[body_info[0]][body_info[1]]["use_shoes"]==False:
            print("use_shoes False , skip")
            continue

        if label == "hair" and base_body_map[body_info[0]][body_info[1]]["use_hair"] == False:
            print("use_hair False , skip")
            continue


        name = "part_" + "%02d" % idx
        dump_dir = os.path.join(dump_root, name)
        Path(dump_dir).mkdir(exist_ok=True, parents=True)

        if label == "shoe":
            dump_path = os.path.join(dump_dir, name )
        else:
            dump_path = os.path.join(dump_dir, name + ".obj")

        warp_lst[dump_dir] = label



        body_key = asset_info["body_key"]  # body key
        if body_key[1] in ["daz", "vroid"]:
            # asset_body = os.path.join( pathlib.Path (part).parent.parent, "smplx_and_offset_smplified.npz" )
            asset_body = key_smpl_map[ asset_key]
        else:
            nake_dir = base_body_map_source[body_key[0]][body_key[1]]["path"]
            asset_body = os.path.join(nake_dir, "smplx_and_offset_smplified.npz")




        # fix bug for texture change
        if not os.path.exists(part):
            asset_visual = ruku_map[asset_key]["Obj_Mesh"]
        else:
            asset_visual = part




        proxy_path = os.path.join( "/aigc_cfs_gdp/Asset/proxy_meshes/", label, asset_key)

        if label == "shoe":

            asset_body_path_lst.append(asset_body)
            asset_body_path_lst.append(asset_body)

            lft_proxy = os.path.join(proxy_path, "left")
            rgt_proxy = os.path.join(proxy_path, "right")
            asset_proxy_path_lst.append(lft_proxy)
            asset_proxy_path_lst.append(rgt_proxy)

            visual_path = pathlib.Path(asset_visual).parent
            left_mesh = os.path.join(visual_path, "left/asset.obj")
            right_mesh = os.path.join(visual_path, "right/asset.obj")
            asset_visual_path_lst.append( left_mesh)
            asset_visual_path_lst.append( right_mesh)

            asset_label_lst.append("l-shoe")
            asset_label_lst.append("r-shoe")

            dump_path_lst.append(dump_path)
            dump_path_lst.append(dump_path)

        else:

            asset_body_path_lst.append( asset_body )
            asset_proxy_path_lst.append( proxy_path )
            asset_visual_path_lst.append( asset_visual )
            asset_label_lst.append(label)
            dump_path_lst.append(dump_path)


    if len( asset_visual) > 0 :


        run_warp(asset_body_path_lst, asset_proxy_path_lst, asset_visual_path_lst, asset_label_lst, target_body_path, lst_name, dump_path_lst,  device)


    json_object = json.dumps(warp_lst, indent=4)
    with open(os.path.join(dump_root, "warp_lst.json"), "w") as f:
        f.write(json_object)

    with open(os.path.join(dump_root, "smplx-path.txt"), "w") as f:
        f.write(f"{target_dir}\n")



if __name__ == '__main__':

    main()

