import json
import bpy
import os
import bmesh
import  numpy as np
from bmesh.types import BMEdge
import pathlib
import os,  sys

def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data

def write_json(fname,j):
    json_object = json.dumps(j, indent=4)
    with open(fname, "w") as outfile:
        outfile.write(json_object)

def sort_edge(edges):
    def nei_check(a, b):
        if a[0] == b [0]:
            a = [a[1], a[0]]
            return True, a, b
        elif a[0] == b [1] :
            a = [a[1], a[0]]
            b = [b[1], b[0]]
            return True, a, b
        elif a[1] == b [0]:
            return True, a, b
        elif a[1] == b[1]:
            b = [b[1], b[0]]
            return True, a, b
        else:
            return False, None, None


    start = edges [0]
    end = edges [-1]
    prev = start
    sorted = [start]
    num_edge = len(edges)

    if num_edge == 1:
        return start

    edges = edges[1:-1]
    while len(edges) > 0:
        ele_left = len(edges)
        for i in range (ele_left):
            is_nei , a, b =  nei_check( prev, edges[i] )
            if is_nei:
                if ele_left == num_edge -2 :
                    sorted[-1] = a
                    sorted.append(b)
                    prev = b
                else:
                    sorted.append(b)
                    prev = b

                edges.pop( i )
                break
    if num_edge > 2 :
        _, a, b = nei_check(sorted[-1], end)
        sorted.append( b )

    elif num_edge == 2 :
        _, a, b = nei_check(sorted[-1], end)
        sorted[-1] = a
        sorted.append( b )

    return sorted


def unselect_all(context):
    me = context.object.data
    bm = bmesh.from_edit_mesh( me )
    # Deselect all elements
    for vert in bm.verts:
        vert.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    # Update the mesh to reflect the changes
    bmesh.update_edit_mesh(me)
    bm.select_history.clear()


def select_from_lst(context, selection_lst):
    unselect_all(context)
    me = context.object.data
    bm = bmesh.from_edit_mesh( me )
    for edge in bm.edges:
        if edge.index in selection_lst:
            edge.select = True
    bmesh.update_edit_mesh(me)
    bm.select_history.clear()



if __name__ == '__main__':

    print ( sort_edge( [[9,1]] ) )