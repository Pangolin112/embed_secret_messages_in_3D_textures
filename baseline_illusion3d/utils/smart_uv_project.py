# ref: https://github.com/lukasHoel/stylemesh/blob/master/scripts/scannet/create_uvs.py

import bpy

import os


def smart_uv_project():
    """
    Smart UV Project for Blender
    """
    # Set paths
    # scene_name = '49a82360aa'
    # scene_name = 'fb5a96b1a2'
    # scene_name = '0cf2e9402d'
    # scene_name = 'e9ac2fc517'
    scene_name = '0e75f3c4d9'
    DATA_DIR = "./data"
    mesh_path = os.path.join(DATA_DIR, f"ScanNetpp/meshes/{scene_name}/mesh_aligned_0.05.ply")

    # 1. clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. load mesh
    bpy.ops.import_mesh.ply(filepath=mesh_path)

    # get obj from blender internal data
    objectList = bpy.data.objects
    obj = objectList[0]
    if obj.type != "MESH":
        raise ValueError("Object is not of type MESH")
    
    # get face info for this obj
    faces = len(obj.data.polygons)

    # set obj as the active one
    bpy.context.view_layer.objects.active = obj

    # 3. reduce face count, decimate
    max_faces = 500000
    if faces > max_faces:
        # do decimation
        ratio = 1.0 * max_faces / faces

        # clean all decimate modifiers
        for m in obj.modifiers:
            if m.type == "DECIMATE":
                bpy.ops.object.modifier_remove(modifier=m)

        print(f"{obj.name} has {faces} faces before decimation")

        # decimate
        modifier = obj.modifiers.new("DecimateMod", 'DECIMATE')
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier="DecimateMod")
        print(f"{obj.name} has {len(obj.data.polygons)} faces after decimation")

        # save decimated mesh
        decimate_mesh_path = os.path.join(DATA_DIR, f"ScanNetpp/meshes/{scene_name}/mesh_decimated.ply")
        bpy.ops.export_mesh.ply(filepath=decimate_mesh_path,
                            use_ascii=False,
                            use_selection=True,
                            use_mesh_modifiers=True,
                            use_normals=False,
                            use_uv_coords=True,
                            use_colors=False)
        
        # reload mesh
        bpy.ops.wm.read_factory_settings(use_empty=True)

        bpy.ops.import_mesh.ply(filepath=decimate_mesh_path)

        # get obj from blender internal data
        objectList = bpy.data.objects
        obj = objectList[0]
        if obj.type != "MESH":
            raise ValueError("Object is not of type MESH")
        
        # get face info for this obj
        faces = len(obj.data.polygons)

        # set obj as the active one
        bpy.context.view_layer.objects.active = obj

    # 4. get uv parameterization
    angle_limit = 1.2217

    # entering edit mode
    bpy.ops.object.editmode_toggle()

    # select all objects elements
    bpy.ops.mesh.select_all(action='SELECT')

    # the actual unwrapping operation, 1.2217 are 70 degrees
    bpy.ops.uv.smart_project(correct_aspect=False, angle_limit=angle_limit)

    # exiting edit mode
    bpy.ops.object.editmode_toggle()

    print("created uv parameterization")

    # 5. save uv mesh
    uv_mesh_path = os.path.join(DATA_DIR, f"ScanNetpp/meshes/{scene_name}/mesh_uv.ply")
    bpy.ops.export_mesh.ply(filepath=uv_mesh_path,
                            use_ascii=False,
                            use_selection=True,
                            use_mesh_modifiers=True,
                            use_normals=False,
                            use_uv_coords=True,
                            use_colors=False)
    
    print("saved uv mesh")


if __name__ == "__main__":
    smart_uv_project()

