import os.path

import trimesh

pt = "/home/rabbityl/tboard/213cc27c-271a-53d1-943f-31ec98fbacdf_k_4/part_02/"

me = trimesh.load( os.path.join( pt, "part_02.obj" ))

parts = trimesh.graph.connected_components(me.vertex_adjacency_graph.edges)

me.merge_close_vertices(eps=0.0001)

me_smooth = trimesh.smoothing.filter_taubin(me, iterations=10)


me_smooth.export( os.path.join( pt, "part_02_smoothed.obj" ) )
