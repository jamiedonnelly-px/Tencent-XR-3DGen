import open3d as o3d
import open3d_example as o3dtut
import numpy as np
import trimesh as tm

# mesh_name = "flower_dress_scaled"
mesh_name = "garment_test_correspondence_shirt_fix"
mesh_path = f"{mesh_name}.obj"
original_mesh = o3d.io.read_triangle_mesh(mesh_path)
print(original_mesh)


processed_mesh = original_mesh.subdivide_loop(number_of_iterations=2)

# processed_mesh = original_mesh.filter_smooth_laplacian(number_of_iterations=10)
# processed_mesh = original_mesh.merge_close_vertices(eps=1e-3)

# original_mesh.compute_vertex_normals()

# print(
#     f'Input mesh has {len(original_mesh.vertices)} vertices and {len(original_mesh.triangles)} triangles'
# )
# o3d.visualization.draw_geometries([original_mesh])

# voxel_size = max(original_mesh.get_max_bound() - original_mesh.get_min_bound()) / 32
# print(f'voxel_size = {voxel_size:e}')
# processed_mesh = original_mesh.simplify_vertex_clustering(
#     voxel_size=voxel_size,
#     contraction=o3d.geometry.SimplificationContraction.Average)
# print(
#     f'Simplified mesh has {len(processed_mesh.vertices)} vertices and {len(processed_mesh.triangles)} triangles'
# )
# # o3d.visualization.draw_geometries([processed_mesh], mesh_show_back_face=True)


def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 1)))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

check_properties("dress ", processed_mesh)
# verts = original_mesh.vertices
# faces = original_mesh.faces

# processed_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# processed_mesh.vertices = o3d.utility.Vector3dVector(verts)
# # garment_ref.vertex_colors = o3d.utility.Vector3dVector(color_ref)
# processed_mesh.triangles = o3d.utility.Vector3iVector(faces)
# processed_mesh.compute_vertex_normals()
# # if viz:
# # o3d.visualization.draw([smplmesh_canonical])
o3d.io.write_triangle_mesh( f"{mesh_name}_processed.obj", processed_mesh )
# print(f"Mesh {mesh_path} Processed.")



# mesh = tm.load_mesh(mesh_path)
# print(mesh)
# meshes = []

# for k in mesh.geometry.keys() :
#     m = mesh.geometry[k]
#     meshes.append( m )

# #check graph connectivity
# meshc = tm.util.concatenate(meshes)
# graph = meshc.vertex_adjacency_graph
# edges = graph.edges
# # check connected component
# labels = tm.graph.connected_component_labels(edges)
# print(labels)