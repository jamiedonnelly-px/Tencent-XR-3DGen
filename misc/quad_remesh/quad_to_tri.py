import os
import json

def parse_obj_line(line):
    """
    Parse a line from an OBJ file and return the face indices along with texture and normal indices if present.

    Parameters:
    line (str): A line from an OBJ file containing face data. f v(/vt/vn) v/vt/vn v/vt/vn v/vt/vn

    Returns:
    list: A list of tuples, each containing the v,vt,vn.
    """
    elements = line.strip().split()
    face_indices = []
    for element in elements[1:]:
        indices = element.split('/')
        vertex_index = indices[0]  
        texture_index = indices[1] if len(indices) > 1 and indices[1] else None
        normal_index = indices[2] if len(indices) > 2 and indices[2] else None
        face_indices.append((vertex_index, texture_index, normal_index))
    return face_indices

def quad_to_triangles(face_indices):
    """
    Convert a quadrilateral represented by four sets of indices into two triangles.

    Parameters:
    face_indices (list): A list containing four tuples, each representing a vertex's index, texture index, and normal index.

    Returns:
    list: A list containing two lists, each representing a triangle's indices.
    """
    assert len(face_indices) == 4, "The input must contain exactly four sets of indices."

    # Create two triangles from the quadrilateral
    triangle1 = face_indices[0:3]
    triangle2 = [face_indices[0], face_indices[2], face_indices[3]]

    return [triangle1, triangle2]

def format_indices(vertex_index, texture_index=None, normal_index=None):
    """
    Format the indices according to their presence.

    Parameters:
    vertex_index (int): The vertex index.
    texture_index (int, optional): The texture index. Defaults to None.
    normal_index (int, optional): The normal index. Defaults to None.

    Returns:
    str: The formatted indices string.
    """
    if texture_index is not None and normal_index is not None:
        return f"{vertex_index}/{texture_index}/{normal_index}"
    elif texture_index is not None:
        return f"{vertex_index}/{texture_index}"
    elif normal_index is not None:
        return f"{vertex_index}//{normal_index}"
    else:
        return str(vertex_index)

def process_obj_file(obj_content):
    """
    Process the content of an OBJ file and convert quadrilateral faces to triangular faces.

    Parameters:
    obj_content (str): The content of the OBJ file as a string.

    Returns:
    str: The processed OBJ content as a string.
    """
    lines = obj_content.split('\n')
    processed_lines = []

    for line in lines:
        if line.startswith('f '):
            face_indices = parse_obj_line(line)
            if len(face_indices) == 4:
                triangles = quad_to_triangles(face_indices)
                for tri in triangles:
                    tri_indices = ' '.join(format_indices(*indices) for indices in tri)
                    processed_lines.append(f"f {tri_indices}")
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    return '\n'.join(processed_lines)

def read_obj_file(file_path):
    with open(file_path, 'r') as file:
        obj_content = file.read()
    return obj_content

def write_obj_file(file_path, obj_content):
    with open(file_path, 'w') as file:
        file.write(obj_content)

def convert_quad_obj_to_tri(input_file, output_file):
    """
    Process the content of an OBJ file and convert quadrilateral faces to triangular faces.

    Parameters:
    input_file (str): input quad obj path string
    output_file (str): output quad obj path string

    Returns:
    str: save flag
    """
    assert os.path.exists(input_file), f"input_file not exists {input_file}"
    obj_content = read_obj_file(input_file)
    processed_obj_content = process_obj_file(obj_content)
    write_obj_file(output_file, processed_obj_content)
    return os.path.exists(processed_obj_content)


def obj_to_json(obj_file_path, json_file_path, y_plus=0):
    """
    convert obj file to model json, for webui vis quad.

    Parameters:
    obj_file_path (str): input quad obj path string
    json_file_path (str): output josn path string

    Returns:
    str: The processed OBJ content as a string.
    Args:
        input_file: _description_
        output_file: _description_
    """
    verts = []
    faces = []

    with open(obj_file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):  # Vertex line
                vertex = list(map(float, line.split()[1:]))
                if y_plus:
                    vertex[1] += y_plus
                verts.append(vertex)
            elif line.startswith('f '):  # Face line
                face_indices = line.split()[1:]
                face = [int(index.split('/')[0]) for index in face_indices]  # Only take the vertex index part
                faces.append(face)
                
    # Convert face indices to zero-based (OBJ is one-based)
    faces = [[index - 1 for index in face] for face in faces]

    data = {
        "verts_count": len(verts),
        "verts": verts,
        "faces_count": len(faces),
        "faces": faces
    }

    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)
    
    return os.path.exists(json_file_path)


if __name__ == "__main__":
    input_file = "/mnt/aigc_bucket_4/pandorax/quad_remesh/1171e2e9-6665-4a14-9dcd-42f28b9e8c38/quad_remesh/quad_mesh.obj"
    output_file = "/mnt/aigc_bucket_4/pandorax/quad_remesh/1171e2e9-6665-4a14-9dcd-42f28b9e8c38/quad_remesh/quad_mesh_tri.obj"
    convert_quad_obj_to_tri(input_file, output_file)
    
    # output_json_file = "/mnt/aigc_bucket_4/pandorax/quad_remesh/1171e2e9-6665-4a14-9dcd-42f28b9e8c38/quad_remesh/quad_mesh.json"
    # obj_to_json(input_file, output_json_file)
