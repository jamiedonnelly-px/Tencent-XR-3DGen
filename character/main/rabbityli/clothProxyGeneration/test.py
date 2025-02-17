import open3d as o3d

m = o3d.io.read_triangle_mesh("/home/rabbityl/Dropbox/skirt_test.obj")

print( m.vertices )

o3d.visualization.draw([m])