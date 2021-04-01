import os
import cv2

from torch.utils.data import Dataset
from utils.data_utils import *


class LINEMOD_Dataset(Dataset):
    def __init__(self, root_dir, cls, bgfolder=None, use_rendered=True):
        self.cls = cls
        self.use_rendered = use_rendered
        self.folder = os.path.join(root_dir, cls)

        if use_rendered:
            self.rendered_folder = os.path.join(root_dir,"rendered",cls)
            self.num_rendered = len(os.listdir(self.rendered_folder)) // 2

            self.bgfolder = bgfolder
            self.bgimg = [os.path.join(bgfolder, img) for img in os.listdir(bgfolder)]

        self.num_real = len(os.listdir(os.path.join(self.folder, 'data'))) // 4

        num_vertices = int(open(os.path.join(self.folder,"OLDmesh.ply")).readlines()[2].split(' ')[-1])
        self.pt_cld = np.loadtxt(os.path.join(self.folder,"OLDmesh.ply"), skiprows=15, max_rows=num_vertices, usecols=(0,1,2)) / 1000.
        self.transform = np.reshape(np.loadtxt(os.path.join(self.folder,"transform.dat"), skiprows=1, usecols=(1,)), [3,4])

        self.pt_cld = (self.transform @ get_homo_coord(self.pt_cld).T).T
        self.bounding_box = get_bouding_box(self.pt_cld)
        self.keypoints = FPSKeypoints(self.pt_cld)

        self.fx = 572.41140
        self.px = 325.26110
        self.fy = 573.57043
        self.py = 242.04899

        self.rendered_intrinsics = np.loadtxt(os.path.join(root_dir,"rendered/camera.txt"))
        self.intrinsics = np.array([[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]])

    def __len__(self):
        if self.use_rendered:
            return self.num_real + self.num_rendered
        else:
            return self.num_real

    def __getitem__(self, index):
        if index < self.num_real:
            image = cv2.imread(os.path.join(self.folder, "data", "color"+str(index)+".jpg"))
            rot = np.loadtxt(os.path.join(self.folder,"data","rot"+str(index)+".rot"), skiprows=1)
            tra = np.loadtxt(os.path.join(self.folder,"data","tra"+str(index)+".tra"), skiprows=1) / 100.

            mask = create_gt_mask(self.pt_cld, self.intrinsics, tra, rot, 255)
            keypoints_2D = project_to_image(self.keypoints, self.intrinsics, tra, rot)
        else:
            image = cv2.imread(os.path.join(self.rendered_folder, "color"+str(index-self.num_real).zfill(5)+".png"))
            bg = cv2.resize(cv2.imread(random.choice(self.bgimg)),(image.shape[1],image.shape[0]))
            rigid = np.loadtxt(os.path.join(self.rendered_folder,"rigid"+str(self.rendered[index-self.num_real]).zfill(5)))
            rot, tra = get_rottra_from_rigid(rigid)

            mask = create_gt_mask(self.pt_cld, self.rendered_intrinsics, tra, rot, 255)
            keypoints_2D = project_to_image(self.keypoints, self.intrinsics, tra, rot)

            image = (image & np.tile(np.expand_dims(mask,axis=2),[1,1,3])) + (bg & np.tile(np.expand_dims(~mask,axis=2),[1,1,3]))

        offset = calc_offset(keypoints_2D)
        image = np.reshape(image, (3,480,640)) / 255.
        sample = (image, mask, offset, keypoints_2D)
        return sample
