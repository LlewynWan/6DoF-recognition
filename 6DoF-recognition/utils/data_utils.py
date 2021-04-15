import cv2
import numpy as np


def get_rigid_transform(tra, rot):
    tra = np.reshape(tra, (3,1))
    rigid_transformation = np.append(rot, tra, axis=1)
    return rigid_transformation


def get_rottra_from_rigid(rigid):
    rot = rigid[0:3,0:3]
    tra = rigid.T[3,0:3]
    return rot, tra


def fill_holes(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    inv_mask = ~mask

    if mask[0][0]==0:
        cv2.floodFill(mask, None, (0,0), 255)
    else:
        size = mask.shape
        cv2.floodFill(mask, None, (size[1]-1, size[0]-1), 255)
    filled_mask = ~(inv_mask & mask)

    return filled_mask


def get_homo_coord(pt_cld):
    ones = np.ones((pt_cld.shape[0],1))
    homo_coord = np.append(pt_cld, ones, axis=1)

    return homo_coord


def project_to_image(pt_cld, intrinsics, tra, rot):
    homo_coord = get_homo_coord(pt_cld)
    rigid = get_rigid_transform(tra,rot)
    homo_coord = intrinsics @ (rigid @ homo_coord.T)

    coord_2D = homo_coord[:2, :] / homo_coord[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)

    return coord_2D


def create_gt_mask(pt_cld, intrinsics, tra, rot, label, image_shape=(480,640)):
    coord_2D = project_to_image(pt_cld, intrinsics, tra, rot)

    ID_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    x_2d = np.clip(coord_2D[:, 0], 0, image_shape[1]-1)
    y_2d = np.clip(coord_2D[:, 1], 0, image_shape[0]-1)
    ID_mask[y_2d, x_2d] = label

    return fill_holes(ID_mask)


def calc_offset(keypoints, image_size=(480,640)):
    numkpt = keypoints.shape[0]
    X,Y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    offset = np.concatenate((np.expand_dims(Y,axis=2),np.expand_dims(X,axis=2)), axis=2)
    offset = np.tile(np.expand_dims(np.transpose(offset,(1,0,2)),axis=2), (1,1,numkpt,1))
    offset = (keypoints - offset) / (image_size[1], image_size[0])

    offset = offset.transpose(2,3,0,1).reshape((len(keypoints)*2,)+image_size)
    
    return offset


def FPSKeypoints(pt_cld, num_keypoints=8):
    center = np.mean(pt_cld, axis=0)
    init_point = pt_cld[np.sum((pt_cld-center)**2,axis=1).argmax()]
    
    keypoints = [init_point,]
    distance = np.sum((pt_cld-init_point)**2,axis=1)
    for i in range(num_keypoints-1):
        new_point = pt_cld[distance.argmax()]
        keypoints.append(new_point)
        distance = np.minimum(distance, np.sum((pt_cld-new_point)**2,axis=1))
   
    keypoints.append(center)
    return np.array(keypoints)


def get_bounding_box(pt_cld):
    x_max, x_min = np.max(pt_cld[:,0]), np.min(pt_cld[:,0])
    y_max, y_min = np.max(pt_cld[:,1]), np.min(pt_cld[:,1])
    z_max, z_min = np.max(pt_cld[:,2]), np.min(pt_cld[:,2])
    bounding_box = np.array([
        [x_min,y_min,z_min],
        [x_max,y_min,z_min],
        [x_max,y_max,z_min],
        [x_min,y_max,z_min],
        [x_min,y_min,z_max],
        [x_max,y_min,z_max],
        [x_max,y_max,z_max],
        [x_min,y_max,z_max]])

    return bounding_box


def draw_bounding_box(bounding_box, image, color=(0,0,0), thickness=2):
    for i in range(4):
        image = cv2.line(image, tuple(bounding_box[i]), tuple(bounding_box[(i+1)%4]), color, thickness)
        image = cv2.line(image, tuple(bounding_box[i+4]), tuple(bounding_box[(i+1)%4+4]), color, thickness)
        image = cv2.line(image, tuple(bounding_box[i]), tuple(bounding_box[i+4]), color, thickness)

    return image
