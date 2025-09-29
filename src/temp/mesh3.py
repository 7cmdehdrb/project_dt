import csv

def write_mesh_to_csv(verts, faces, vfilename='verts.csv', ffilename='faces.csv'):
    """
    Writes verts and faces arrays to separate CSV files.

    Parameters:
        verts (ndarray): (n, 3) array of vertex coordinates.
        faces (ndarray): (n, 3) array of face indices.
        vfilename (str): Output CSV file name for vertices.
        ffilename (str): Output CSV file name for faces.
    """
    with open(vfilename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header for vertices
        writer.writerow(['verts_x', 'verts_y', 'verts_z'])
        for vert in verts:
            writer.writerow(vert)

    with open(ffilename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header for faces
        writer.writerow(['faces_v1', 'faces_v2', 'faces_v3'])
        for face in faces:
            writer.writerow(face)
            
    print(f"Wrote {len(verts)} vertices to {vfilename} and {len(faces)} faces to {ffilename}.")
    
    
    write_mesh_to_csv(verts, faces, vfilename='/home/min/7cmdehdrb/project_th/verts.csv', ffilename='/home/min/7cmdehdrb/project_th/faces.csv')