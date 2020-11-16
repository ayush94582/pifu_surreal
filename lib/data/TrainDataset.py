from torch.utils.data import Dataset
import numpy as np
import os
import glob
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging



log = logging.getLogger('trimesh')
#log.setLevel(40)

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    random.shuffle(folders)
    meshs = {}
    count_frames = 0
    for i, f in enumerate(folders):
        sub_name = f
        frames_list = glob.glob(os.path.join(root_dir, f) + "/*/frame*.obj")
        meshs[sub_name] = {}
        for frame_obj in frames_list:
            count_frames += 1
            frame_num = int(frame_obj.split("frame")[1].split(".obj")[0])
            meshs[sub_name][frame_num] = frame_obj
    print("TOTAL MESHES")
    print(len(meshs))
    print("COUNT FRAMES")
    print(count_frames)
    return meshs, count_frames

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

        self.B_MIN = np.array([-5, -5, -5])#[-128, -28, -128])
        self.B_MAX = np.array([5, 5, 5])#[128, 228, 128])
        #self.B_MIN = np.array([-128, -28, -128])
        #self.B_MAX = np.array([128, 228, 128])
        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize
        self.temporal_length = self.opt.temporalSize

        self.num_views = self.opt.num_views

        self.num_sample_inout =  1000
        print(f"NUM SAMPLE IN OUT: {self.num_sample_inout}")
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0,360,45))
        #self.yaw_list = [0]
        self.pitch_list = [0]

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        #frame_limit = 1000 if self.is_train else 400
        self.mesh_dic, self.frame_count = load_trimesh(self.OBJ)
        self.subjects = self.get_subjects()

    def get_subjects(self):
        all_subjects = sorted(list(self.mesh_dic.keys())) #os.listdir(self.RENDER)

        if self.is_train:
            return all_subjects[:-100]
        else:
            return all_subjects[-100:]

    def __len__(self):
        return len(self.subjects) #len(self.subjects)

    def get_render(self, subject, vid_name, num_views, fid, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        pitch = self.pitch_list[pid]

        # The ids are an even distribution of num_views around view_id
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        for vid in view_ids:
            param_path = os.path.join(self.PARAM, subject, vid_name, 'frame%d_%d_%d_%02d.npy' % (fid, vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, vid_name, 'frame%d_%d_%d_%02d.jpg' % (fid, vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, vid_name, 'frame%d_%d_%d_%02d.png' % (fid, vid, pitch, 0))

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                #render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0)
        }

    def select_sampling_method(self, subject, vid_name, fid):
        mesh = trimesh.load(self.mesh_dic[subject][fid])
        #mesh.vertices -= mesh.center_mass
        #mesh = mesh.apply_scale(30.0)
        #mesh = mesh.apply_translation((0,100,0))
        #mesh = mesh.apply_transform(trimesh.transformations.rotation_matrix(90, (0,-1,0), point=None))
        #bb = trimesh.bounds.corners(mesh.bounding_box_oriented.bounds)
        #print(bb)

        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)
        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]                                                                               

        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        #print(f"Subject: {subject}, FID: {fid}")
        #save_samples_truncted_prob('out.ply', samples.T, labels.T)
        #exit()

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        
        del mesh

        return {
            'samples': samples,
            'labels': labels,
        }


    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        sid = index % len(self.subjects)
        yid = random.choice(list(range(len(self.yaw_list)))) #tmp % len(self.yaw_list)
        pid = 0 # tmp // len(self.pitch_list)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        vid_name = os.listdir(os.path.join(self.RENDER, subject))[0]
        #frame_paths = os.listdir(os.path.join(self.RENDER, subject, vid_name))
        frame_ids = [int(frame_id) for frame_id in list(self.mesh_dic[subject].keys())]

        #frame_ids = list(set([int(frame_path.split("frame")[1].split("_")[0]) for frame_path in frame_paths]))

        fid = random.choice(frame_ids) 


        res = {
            'name': subject,
            'video_name': vid_name,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'fid': fid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, vid_name, num_views=self.num_views, yid=yid, pid=pid, fid=fid,
                                        random_sample=self.opt.random_multiview)
                             
        res.update(render_data)

        sample_data = self.select_sampling_method(subject,vid_name,fid)
        res.update(sample_data)

        return res


    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            return None
