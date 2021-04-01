import bpy
import bmesh
import bpy_extras

import os
import random
import numpy as np

import math
import mathutils
from mathutils import Matrix
from mathutils import Vector


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for Tonstraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = (R_bcam2cv@T_world2bcam) / 1000.

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


def look_at(obj_camera, point, rand_roll=True):
    loc_camera = obj_camera.location
    #matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
   
    if rand_roll:
        roll = math.radians(random.randint(0,360))
        camera_roll = mathutils.Matrix.Rotation(roll, 4, 'Z')
        obj_camera.rotation_euler = (rot_quat.to_matrix().to_4x4() @ camera_roll).to_euler()
    else:
        obj_camera.rotation_euler = rot_quat.to_euler()


def randpos(center, sun=False):
    altitude = center[-1]
    Z = altitude + random.randint(-500,500)
    radius = random.randint(700, 1500)
    angle = 2*math.pi*random.random()
    X = radius * math.cos(angle)
    Y = radius * math.sin(angle)
    
    return (X, Y, Z)


def randview(center, diameter):
    X,Y,Z = center
    X += (random.random()*1.5-0.75) * diameter
    Y += (random.random()*1.5-0.75) * diameter
    Z += (random.random()*1.5-0.75) * diameter

    return (X,Y,Z)


voc2012_path = "/media/llewyn/TOSHIBA EXT/PASCAL_VOC_2012/VOCdevkit/VOC2012/JPEGImages/"
voc_images = os.listdir(voc2012_path)

linemod_path = "/media/llewyn/TOSHIBA EXT/LINEMOD/LINEMOD/"
linemod_classes = ['ape', 'benchviseblue', 'cam', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']


# ----------------------------------------------------------
if __name__ == "__main__":
    context = bpy.context
    scene = context.scene
    
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    bpy.data.objects['Light'].select_set(True)
    bpy.ops.object.delete()
    bpy.context.scene.render.film_transparent = False

    cam = bpy.data.objects['Camera']
    cam.data.clip_end = 100000.

    img = bpy.data.images.load(os.path.join(voc2012_path, '2011_002985.jpg')) 
    
    scene.render.engine = 'CYCLES'
    scene.render.film_transparent = True
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.image_settings.file_format = 'PNG'
    
    bpy.ops.material.new()
    material = list(bpy.data.materials)[0]

    material.use_nodes = True
    material.node_tree.links.clear()

    mat_out = material.node_tree.nodes['Material Output']
    diffuse_node = material.node_tree.nodes['Principled BSDF']
    gloss_node = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    attr_node = material.node_tree.nodes.new(type='ShaderNodeAttribute')

    material.node_tree.nodes.remove(diffuse_node)
    attr_node.attribute_name = 'Col'
    material.node_tree.links.new(attr_node.outputs['Color'], gloss_node.inputs['Base Color'])
    material.node_tree.links.new(gloss_node.outputs['BSDF'], mat_out.inputs['Surface'])

    light_data = bpy.data.lights.new(name='Light', type='POINT')
    light_data.distance = 1000.
    light_data.cutoff_distance = 1700.
    light_data.use_nodes = True

    for cls in linemod_classes:
        bpy.ops.import_mesh.ply(filepath=linemod_path+cls+"/OLDmesh.ply")
        diameter = float(open(linemod_path+cls+"/distance.txt").read()) * 10
        transform = np.loadtxt(linemod_path+cls+"/transform.dat", skiprows=1, usecols=(1,))
        transform = np.append(np.reshape(transform, [3,4]), np.array([[0,0,0,1]]), axis=0)
        transform[:-1,3] *= 1000.

        obj = bpy.data.objects['OLDmesh']
        obj.data.transform(mathutils.Matrix(transform))
        obj.data.update()

        mverts_co = np.zeros((len(obj.data.vertices)*3), dtype=np.float)
        obj.data.vertices.foreach_get("co",mverts_co)
        mverts_co = np.reshape(mverts_co, [len(obj.data.vertices),3])
        center = np.mean(mverts_co, axis=0)

        obj.select_set(False)

        for i in range(5):
            cam.location = randpos(center)
            look_at(cam, Vector(randview(center, diameter)))

            scene.camera = cam 

            light_strength = random.randint(700, 2500)
            light_data.energy = light_strength
            light_data.node_tree.nodes['Emission'].inputs['Strength'].default_value = light_strength

            for j in range(12):
                light = bpy.data.objects.new(name='Light'+str(j), object_data=light_data)
                bpy.context.collection.objects.link(light)
                bpy.context.view_layer.objects.active = light
                light.location = randpos(center)

            obj.data.materials.append(material)
            scene.render.filepath = linemod_path+"rendered/"+cls+"/"+str(i).zfill(5)
            bpy.ops.render.render(write_still=True, use_viewport=False)

            P, K, RT = get_3x4_P_matrix_from_blender(cam)
            np.savetxt(linemod_path+"rendered/"+cls+"/rigid"+str(i).zfill(5),RT)

            for j in range(12):
                bpy.data.objects['Light'+str(j)].select_set(True)
                bpy.ops.object.delete()
        
        bpy.data.objects['OLDmesh'].select_set(True)
        bpy.ops.object.delete()
