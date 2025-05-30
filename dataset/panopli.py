# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import os
import math
import random
import cv2
import h5py
from pathlib import Path

import torch
import pickle
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from torchvision.utils import save_image

from dataset.base import create_segmentation_data_base, BaseDataset, process_bounding_box_dict
from dataset.preprocessing.preprocess_scannet import get_thing_semantics
from util.camera import compute_world2normscene
from util.misc import EasyDict
from util.ray import get_ray_directions_with_intrinsics, get_rays, rays_intersect_sphere


class PanopLiDataset(BaseDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, load_depth=False, load_feat=False, semantics_dir="filtered_semantics", instance_dir='filtered_instance', 
                 instance_to_semantic_key='instance_to_semantic', create_seg_data_func=create_segmentation_data_base, subsample_frames=1):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, load_depth, load_feat, semantics_dir, instance_dir, instance_to_semantic_key, create_seg_data_func, subsample_frames, False)
        self.faulty_classes = [0]
        self.is_thing = get_thing_semantics()
        self.all_frame_names = []
        self.all_probabilities, self.all_confidences = [], []
        self.all_origins = []
        self.all_feats = []
        self.world2scene = np.eye(4, dtype=np.float32)
        self.force_reset_fov = False
        self.full_train_set_mode = True
        self.random_train_val_ratio = 0.90
        self.setup_data()

    def setup_data(self):
        self.all_frame_names = sorted([x.stem for x in (self.root_dir / "color").iterdir() if x.name.endswith('.jpg')], key=lambda y: int(y) if y.isnumeric() else y)
        sample_indices = list(range(len(self.all_frame_names)))
        if self.overfit:
            self.train_indices = self.val_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            self.train_indices = self.train_indices * 5
        else:
            if (self.root_dir / "splits.json").exists():
                #print(self.root_dir)
                split_json = json.loads((self.root_dir / "splits.json").read_text())
                self.train_indices = [self.all_frame_names.index(f'{x}') for x in split_json['train']]
                if self.split == "test":
                    if 'test' in split_json:
                        self.val_indices = [self.all_frame_names.index(f'{x}') for x in split_json['test']]
                    else:  # itw has no labels for evaluation, so it doesn't matter
                        self.val_indices = [self.all_frame_names.index(f'{x}') for x in split_json['val']]
                else:
                    if self.full_train_set_mode:  # for final training
                        self.val_indices = [self.all_frame_names.index(f'{x}') for x in split_json['test']]
                    else:  # random val set
                        train_names = random.sample(split_json['train'], int(self.random_train_val_ratio * len(split_json['train'])))
                        val_names = [x for x in split_json['train'] if x not in train_names]
                        self.train_indices = [self.all_frame_names.index(f'{x}') for x in train_names]
                        self.val_indices = [self.all_frame_names.index(f'{x}') for x in val_names]
                self.num_val_samples = len(self.val_indices)
            else:
                self.val_indices = np.random.choice(sample_indices, min(len(self.all_frame_names), self.num_val_samples))
                self.train_indices = [sample_index for sample_index in sample_indices if sample_index not in self.val_indices]
        self.train_indices = self.train_indices[::self.subsample_frames]
        self.val_indices = self.val_indices[::self.subsample_frames]
        dims, intrinsics, cam2scene = [], [], []
        img_h, img_w = np.array(Image.open(self.root_dir / "color" / f"{self.all_frame_names[0]}.jpg")).shape[:2]
        for sample_index in sample_indices:
            intrinsic_color = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(self.root_dir / "intrinsic" / "intrinsic_color.txt").read_text().splitlines() if x != ''])
            intrinsic_color = intrinsic_color[:3, :3]
            if self.force_reset_fov:
                intrinsic_color[0, 0] = intrinsic_color[0, 2] / math.tan(math.radians(90) / 2)
                intrinsic_color[1, 1] = intrinsic_color[1, 2] / math.tan(math.radians(90) / 2)
            cam2world = np.array([[float(y.strip()) for y in x.strip().split()] for x in Path(self.root_dir / "pose" / f"{self.all_frame_names[sample_index]}.txt").read_text().splitlines() if x != ''])
            cam2scene.append(torch.from_numpy(self.world2scene @ cam2world).float())
            self.cam2scenes[sample_index] = cam2scene[-1]
            dims.append([img_h, img_w])
            intrinsics.append(torch.from_numpy(intrinsic_color).float())
            self.intrinsics[sample_index] = intrinsic_color
            self.intrinsics[sample_index] = torch.from_numpy(np.diag([self.image_dim[1] / img_w
                                                                         , self.image_dim[0] / img_h, 1]) @ self.intrinsics[sample_index]).float()
        self.scene2normscene = compute_world2normscene(
            torch.Tensor(dims).float(),
            torch.stack(intrinsics).float(),
            torch.stack(cam2scene).float(),
            max_depth=self.max_depth,
            rescale_factor=1.0
        )

        self.normscene_scale = self.scene2normscene[0, 0]
        for sample_index in sample_indices:
            self.cam2normscene[sample_index] = self.scene2normscene @ self.cam2scenes[sample_index]

        if self.split == "train":
            for sample_index in tqdm(self.train_indices, desc='dataload'):
                image, rays, semantics, instances, depth, _, probabilities, confidences, feat, room_mask, pose = self.load_sample(sample_index)
                self.all_rgbs.append(image)
                self.all_rays.append(rays)
                self.all_semantics.append(semantics)
                self.all_probabilities.append(probabilities)
                self.all_confidences.append(confidences)
                self.all_instances.append(instances)
                self.all_masks.append(room_mask)
                if self.load_feat:
                    self.all_feats.append(feat)
                if self.load_depth:
                    self.all_depths.append(depth)
                self.all_origins.append(torch.ones_like(semantics) * sample_index)
                self.all_poses.append(pose)

        pkl_segmentation_data = pickle.load(open(self.root_dir / 'segmentation_data.pkl', 'rb'))
        self.segmentation_data = self.create_segmentation_data(self, pkl_segmentation_data)

    def load_sample(self, sample_index):
        cam2normscene = self.cam2normscene[sample_index]
        image = Image.open(self.root_dir / "color" / f"{self.all_frame_names[sample_index]}.jpg")
        # noinspection PyTypeChecker
        image = torch.from_numpy(np.array(image.resize(self.image_dim[::-1], Image.LANCZOS)) / 255).float()
        semantics = Image.open(self.root_dir / self.semantics_directory / f"{self.all_frame_names[sample_index]}.png")
        instances = Image.open(self.root_dir / self.instance_directory / f"{self.all_frame_names[sample_index]}.png")
        # noinspection PyTypeChecker
        semantics = torch.from_numpy(np.array(semantics.resize(self.image_dim[::-1], Image.NEAREST))).long()
        npz = np.load(self.root_dir / f"{self.semantics_directory.split('_')[0]}_probabilities" / f"{self.all_frame_names[sample_index]}.npz")
        probabilities, confidences = torch.from_numpy(npz['probability']), torch.from_numpy(npz['confidence'])
        if "notta" in self.semantics_directory and 'confidence_notta' in npz:
            confidences = torch.from_numpy(npz['confidence_notta'])
        elif "notta" in self.semantics_directory and 'confidence_notta' not in npz:
            confidences = torch.ones_like(confidences)
            print("WARNING: Confidences not found in npz")
        interpolated_p = torch.nn.functional.interpolate(torch.cat([probabilities.permute((2, 0, 1)), confidences.unsqueeze(0)], 0).unsqueeze(0), size=self.image_dim[::-1], mode='bilinear', align_corners=False).squeeze(0)
        probabilities, confidences = interpolated_p[:-1, :, :].permute((1, 2, 0)).cpu(), interpolated_p[-1, :, :].cpu()
        # noinspection PyTypeChecker
        feat = torch.zeros(1)
        if self.load_feat:
            npz = np.load(self.root_dir / f"{self.semantics_directory.split('_')[0]}_feats" / f"{self.all_frame_names[sample_index]}.npz")
            feat = torch.nn.functional.interpolate(torch.from_numpy(npz["feats"]).permute((2, 0, 1)).unsqueeze(0), size=self.image_dim[::-1], mode='bilinear', align_corners=False).squeeze(0).permute((1, 2, 0))

        instances = torch.from_numpy(np.array(instances.resize(self.image_dim[::-1], Image.NEAREST))).long()
        # noinspection PyTypeChecker
        depth = torch.zeros(1)
        depth_cam = torch.zeros(1)
        if self.load_depth:
            raw_depth = np.array(Image.open(self.root_dir / "depth" / f"{self.all_frame_names[sample_index]}.png"))
            raw_depth = raw_depth.astype(np.float32) / 1000
            raw_depth[raw_depth > (self.max_depth / self.normscene_scale.item())] = (self.max_depth / self.normscene_scale.item())
            # noinspection PyTypeChecker
            depth_cam = torch.from_numpy(np.array(Image.fromarray(raw_depth).resize(self.image_dim[::-1], Image.NEAREST)))
            depth_cam_s = self.normscene_scale * depth_cam
            depth = depth_cam_s.float()

        directions = get_ray_directions_with_intrinsics(self.image_dim[0], self.image_dim[1], self.intrinsics[sample_index].numpy())
        # directions = get_ray_directions_with_intrinsics_undistorted(self.image_dim[0], self.image_dim[1], self.intrinsics[sample_index].numpy(), self.distortion_params)
        rays_o, rays_d = get_rays(directions, cam2normscene)

        sphere_intersection_displacement = rays_intersect_sphere(rays_o, rays_d, r=1)  # fg is in unit sphere

        rays = torch.cat(
            [rays_o, rays_d, 0.01 *
             torch.ones_like(rays_o[:, :1]), sphere_intersection_displacement[:, None], ], 1,
        )
        room_mask_path = self.root_dir / "invalid" / f"{self.all_frame_names[sample_index]}.jpg"
        if room_mask_path.exists():
            room_mask = ~torch.from_numpy(np.array(Image.open(room_mask_path).resize(self.image_dim[::-1], Image.NEAREST)) > 0).bool()
        else:
            room_mask = torch.ones(rays.shape[0]).bool()
        return image.reshape(-1, 3), rays, semantics.reshape(-1), instances.reshape(-1), depth.reshape(-1),\
               depth_cam.reshape(-1), probabilities.reshape(-1, probabilities.shape[-1]), confidences.reshape(-1),\
               feat.reshape(-1, feat.shape[-1]), room_mask.reshape(-1), cam2normscene

    def export_point_cloud(self, output_path, subsample=1, export_semantics=False, export_bbox=False):
        super().export_point_cloud(output_path, subsample, export_semantics, export_bbox)

    @property
    def num_instances(self):
        return self.segmentation_data.num_instances

    @property
    def things_filtered(self):
        return set([i for i in range(len(self.is_thing)) if self.is_thing[i]]) - set(self.faulty_classes)

    @property
    def stuff_filtered(self):
        return set([i for i in range(len(self.is_thing)) if not self.is_thing[i]]) - set(self.faulty_classes)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.split == 'val' or self.split == 'test':
            sample_idx = self.val_indices[idx % len(self.val_indices)]
            semantics = Image.open(self.root_dir / "rs_semantics" / f"{self.all_frame_names[sample_idx]}.png")
            instances = Image.open(self.root_dir / "rs_instance" / f"{self.all_frame_names[sample_idx]}.png")
            semantics = torch.from_numpy(np.array(semantics.resize(self.image_dim[::-1], Image.NEAREST))).long().reshape(-1)
            instances = torch.from_numpy(np.array(instances.resize(self.image_dim[::-1], Image.NEAREST))).long().reshape(-1)
            sample['rs_semantics'] = semantics
            sample['rs_instances'] = instances
        return sample


class InconsistentPanopLiDataset(PanopLiDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, semantic_class, overfit=False, num_val_samples=8, max_rays=512, semantics_dir='filtered_semantics',
                 instance_dir='filtered_instance_inc', instance_to_semantic_key='instance_to_semantic_inc', create_seg_data_func=create_segmentation_data_base):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, semantics_dir=semantics_dir, instance_dir=instance_dir,
                         instance_to_semantic_key=instance_to_semantic_key, create_seg_data_func=create_seg_data_func)
        print('Preparing InconsistentPanopLiDataset...')
        all_rays = []
        all_instances = []
        for i in range(len(self.train_indices)):
            all_semantics_view = self.all_semantics[self.all_origins == self.train_indices[i]]
            all_instances_view = self.all_instances[self.all_origins == self.train_indices[i]]
            all_rays_view = self.all_rays[self.all_origins == self.train_indices[i], :]
            mask = all_semantics_view == semantic_class
            if mask.sum() > 0:
                all_rays.append(all_rays_view[mask, :])
                all_instances.append(all_instances_view[mask])
        self.all_rays = all_rays
        self.all_instances = all_instances
        self.max_rays = max_rays

    def __getitem__(self, idx):
        selected_rays = self.all_rays[idx]
        selected_instances = self.all_instances[idx]
        if selected_rays.shape[0] > self.max_rays:
            sampled_indices = random.sample(range(selected_rays.shape[0]), self.max_rays)
            selected_rays = selected_rays[sampled_indices, :]
            selected_instances = selected_instances[sampled_indices]
        sample = {
            f"rays": selected_rays,
            f"instances": selected_instances,
        }
        return sample

    def __len__(self):
        return len(self.all_rays)

    @staticmethod
    def collate_fn(batch):
        return {
            "rays": [x["rays"] for x in batch],
            "instances": [x["instances"] for x in batch],
        }


class InconsistentPanopLiSingleDataset(PanopLiDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, max_rays=512, semantics_dir='filtered_semantics',
                 instance_dir='filtered_instance_inc', instance_to_semantic_key='instance_to_semantic_inc', create_seg_data_func=create_segmentation_data_base, subsample_frames=1):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, semantics_dir=semantics_dir,
                         instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key, create_seg_data_func=create_seg_data_func, subsample_frames=subsample_frames)
        print('Preparing InconsistentPanopLiDataset...')
        #all_rays_view = self.all_rays.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1], -1)
        #all_instances_view = self.all_instances.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        #all_confidences_view = self.all_confidences.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        #all_masks_view = self.all_masks.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        #all_confidences_view[~all_masks_view] = 0
        all_rays = []
        all_instances = []
        all_confidences = []
        for i in range(len(self.train_indices)):
            mask = self.all_instances[i] != 0
            if mask.sum() > 0:
                all_rays.append(self.all_rays[i][mask, :])
                all_instances.append(self.all_instances[i][mask])
                confidences = self.all_confidences[i]
                room_mask = self.all_masks[i]
                confidences[~room_mask] = 0
                all_confidences.append(confidences[mask])
        self.all_rays = all_rays
        self.all_instances = all_instances
        self.all_confidences = all_confidences
        self.max_rays = max_rays

    def __getitem__(self, idx):
        selected_rays = self.all_rays[idx]
        selected_instances = self.all_instances[idx]
        selected_confidences = self.all_confidences[idx]
        if selected_rays.shape[0] > self.max_rays:
            #sampled_indices = random.sample(range(selected_rays.shape[0]), self.max_rays)
            sampled_indices = torch.randperm(selected_rays.shape[0])[:self.max_rays]
            selected_rays = selected_rays[sampled_indices, :]
            selected_instances = selected_instances[sampled_indices]
            selected_confidences = selected_confidences[sampled_indices]
        sample = {
            f"rays": selected_rays,
            f"instances": selected_instances,
            f"confidences": selected_confidences,
        }
        return sample

    def __len__(self):
        return len(self.all_rays)

    @staticmethod
    def collate_fn(batch):
        return {
            "rays": [x["rays"] for x in batch],
            "instances": [x["instances"] for x in batch],
            "confidences": [x["confidences"] for x in batch],
        }


def create_segmentation_data_panopli(dataset_ref, seg_data):
    seg_data_dict = EasyDict({
        'fg_classes': sorted(seg_data['fg_classes']),
        'bg_classes': sorted(seg_data['bg_classes']),
        'instance_to_semantics': seg_data[dataset_ref.instance_to_semantic_key],
        'num_semantic_classes': len(seg_data['fg_classes'] + seg_data['bg_classes']),
        'num_instances': len(seg_data['fg_classes'])
    })
    return seg_data_dict


def create_segmentation_data_panopli_with_valid(dataset_ref, seg_data):
    seg_data_dict = create_segmentation_data_panopli(dataset_ref, seg_data)
    seg_data_dict['instance_is_valid'] = seg_data['m2f_sem_valid_instance']
    return seg_data_dict


def create_segmentation_data_panopli_mmdet(dataset_ref, seg_data):
    dataset_ref.bounding_boxes = process_bounding_box_dict(seg_data['mmdet_bboxes'], dataset_ref.scene2normscene.numpy())
    return EasyDict({
        'fg_classes': sorted(seg_data['fg_classes']),
        'bg_classes': sorted(seg_data['bg_classes']),
        'instance_to_semantics': {
            (i + 1): seg_data['mmdet_bboxes'][i]['class']
            for i in range(dataset_ref.bounding_boxes.ids.shape[0])
        },
        'num_semantic_classes': len(seg_data['fg_classes'] + seg_data['bg_classes']),
        'num_instances': dataset_ref.bounding_boxes.ids.shape[0]
    })


def create_segmentation_data_panopli_gt(dataset_ref, seg_data):
    dataset_ref.bounding_boxes = process_bounding_box_dict(seg_data['gt_bboxes'], dataset_ref.scene2normscene.numpy())
    return EasyDict({
        'fg_classes': sorted(seg_data['fg_classes']),
        'bg_classes': sorted(seg_data['bg_classes']),
        'instance_to_semantics': {
            (i + 1): seg_data['gt_bboxes'][i]['class']
            for i in range(dataset_ref.bounding_boxes.ids.shape[0])
        },
        'num_semantic_classes': len(seg_data['fg_classes'] + seg_data['bg_classes']),
        'num_instances': dataset_ref.bounding_boxes.ids.shape[0]
    })


class SegmentPanopLiDataset(PanopLiDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, max_rays=512, semantics_dir='filtered_semantics',
                 instance_dir='filtered_instance_inc', instance_to_semantic_key='instance_to_semantic_inc', create_seg_data_func=create_segmentation_data_base, subsample_frames=1):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, semantics_dir=semantics_dir,
                         instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key, create_seg_data_func=create_seg_data_func, subsample_frames=subsample_frames)
        print('Preparing SegmentPanopLi...')
        #all_rays_view = self.all_rays.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1], -1)
        #all_confidences_view = self.all_confidences.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        #all_masks_view = self.all_masks.view(len(self.train_indices), self.image_dim[0] * self.image_dim[1])
        #all_confidences_view[~all_masks_view] = 0
        all_rays = []
        all_confidences = []
        all_ones = []
        for i in range(len(self.train_indices)):
            segments = Image.open(self.root_dir / "m2f_segments" / f"{self.all_frame_names[self.train_indices[i]]}.png")
            segments = torch.from_numpy(np.array(segments.resize(self.image_dim[::-1], Image.NEAREST))).long().reshape(-1)
            for s in torch.unique(segments):
                if s.item() != 0:
                    all_rays.append(self.all_rays[i][segments == s, :])
                    confidences = self.all_confidences[i]
                    room_mask = self.all_masks[i]
                    confidences[~room_mask] = 0
                    all_confidences.append(confidences[segments == s])
                    all_ones.append(torch.ones(all_confidences[-1].shape[0]).long())
        self.all_rays = all_rays
        self.all_confidences = all_confidences
        self.all_ones = all_ones
        self.max_rays = max_rays
        self.enabled = False

    def __getitem__(self, idx):
        if self.enabled:
            selected_rays = self.all_rays[idx]
            selected_confidences = self.all_confidences[idx]
            selected_ones = self.all_ones[idx]
            if selected_rays.shape[0] > self.max_rays:
                #sampled_indices = random.sample(range(selected_rays.shape[0]), self.max_rays)
                sampled_indices = torch.randperm(selected_rays.shape[0])[:self.max_rays]
                selected_rays = selected_rays[sampled_indices, :]
                selected_confidences = selected_confidences[sampled_indices]
                selected_ones = selected_ones[sampled_indices]
            sample = {
                f"rays": selected_rays,
                f"confidences": selected_confidences,
                f"group": selected_ones,
            }
        else:
            sample = {
                f"rays": [0],
                f"confidences": [0],
                f"group": [0],
            }
        return sample

    def __len__(self):
        return len(self.all_rays)

    @staticmethod
    def collate_fn(batch):
        return {
            "rays": [x["rays"] for x in batch],
            "confidences": [x["confidences"] for x in batch],
            "group": [batch[i]['group'] * i for i in range(len(batch))]
        }


class DistinctColors:

    def __init__(self):
        colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f55031', '#911eb4', '#42d4f4', '#bfef45', '#fabed4', '#469990',
            '#dcb1ff', '#404E55', '#fffac8', '#809900', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#f032e6',
            '#806020', '#ffffff',

            "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0030ED", "#3A2465", "#34362D", "#B4A8BD", "#0086AA",
            "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700",

            "#04F757", "#C8A1A1", "#1E6E00",
            "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
            "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        ]
        self.hex_colors = colors
        # 0 = crimson / red, 1 = green, 2 = yellow, 3 = blue
        # 4 = orange, 5 = purple, 6 = sky blue, 7 = lime green
        self.colors = [hex_to_rgb(c) for c in colors]
        self.color_assignments = {}
        self.color_ctr = 0
        self.fast_color_index = torch.from_numpy(np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))

    def get_color(self, index, override_color_0=False):
        colors = [x for x in self.hex_colors]
        if override_color_0:
            colors[0] = "#3f3f3f"
        colors = [hex_to_rgb(c) for c in colors]
        if index not in self.color_assignments:
            self.color_assignments[index] = colors[self.color_ctr % len(self.colors)]
            self.color_ctr += 1
        return self.color_assignments[index]

    def get_color_fast_torch(self, index):
        return self.fast_color_index[index]

    def get_color_fast_numpy(self, index, override_color_0=False):
        index = np.array(index).astype(np.int32)
        if override_color_0:
            colors = [x for x in self.hex_colors]
            colors[0] = "#3f3f3f"
            fast_color_index = torch.from_numpy(np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))
            return fast_color_index[index % fast_color_index.shape[0]].numpy()
        else:
            return self.fast_color_index[index % self.fast_color_index.shape[0]].numpy()

    def apply_colors(self, arr):
        out_arr = torch.zeros([arr.shape[0], 3])

        for i in range(arr.shape[0]):
            out_arr[i, :] = torch.tensor(self.get_color(arr[i].item()))
        return out_arr

    def apply_colors_fast_torch(self, arr):
        return self.fast_color_index[arr % self.fast_color_index.shape[0]]

    def apply_colors_fast_numpy(self, arr):
        return self.fast_color_index.numpy()[arr % self.fast_color_index.shape[0]]
        
def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) / 255 for i in (1, 3, 5)]        
    
class CorresPanopLiDataset(PanopLiDataset):

    def __init__(self, root_dir, split, image_dim, max_depth, overfit=False, num_val_samples=8, max_rays=1024, semantics_dir='filtered_semantics',
                 instance_dir='filtered_instance_inc', instance_to_semantic_key='instance_to_semantic_inc', create_seg_data_func=create_segmentation_data_base, subsample_frames=1):
        super().__init__(root_dir, split, image_dim, max_depth, overfit, num_val_samples, semantics_dir=semantics_dir,
                         instance_dir=instance_dir, instance_to_semantic_key=instance_to_semantic_key, create_seg_data_func=create_seg_data_func, subsample_frames=subsample_frames)
        print('Preparing CorresPanopLiDataset...')
        self.max_rays = max_rays
        all_corres_maps = []
        all_conf_maps = []
        all_mask_valid_corr = []
        all_corres_idx = []
        for i in range(len(self.train_indices)):
            h5_file = self.root_dir / "m2f_correspondence_10" / f"{self.all_frame_names[self.train_indices[i]]}.h5"
            f = h5py.File(h5_file, 'r')
            corres_maps = np.array(f['corres_maps'])
            conf_maps = np.array(f['conf_maps'])
            mask_valid_corr = np.array(f['mask_valid_corr'])
            corres_idx = np.array(f['corres_idx'])
        
            corres_maps = torch.from_numpy(corres_maps)
            conf_maps = torch.from_numpy(conf_maps)
            mask_valid_corr = torch.from_numpy(mask_valid_corr)
            #corres_idx = torch.from_numpy(corres_idx)
            
            all_corres_maps.append(corres_maps)
            all_conf_maps.append(conf_maps)
            all_mask_valid_corr.append(mask_valid_corr)
            all_corres_idx.append(corres_idx)
            
        self.all_corres_maps = all_corres_maps
        self.all_conf_maps = all_conf_maps
        self.all_mask_valid_corr = all_mask_valid_corr
        self.all_corres_idx = all_corres_idx
        #self.all_cam2normscene = [self.cam2normscene[i] for i in self.train_indices]  
        self.all_intrinsics = [self.intrinsics[i] for i in self.train_indices]  #self.intrinsics[self.train_indices]
        
        H, W = self.image_dim[0], self.image_dim[1]
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        self.grid = torch.stack((xx, yy), dim=-1).float()  # ( H, W, 2)
        self.grid_flat = self.grid[:, :, 1] * W + self.grid[:, :, 0]  # (H, W), corresponds to index in flattedned array (in H*W)
        self.grid_flat = self.grid_flat.long()

    def __getitem__(self, idx):
        #self_rgbs_view = self.all_rgbs[idx]
        self_rays_view = self.all_rays[idx]
        self_confidences_view = self.all_confidences[idx]
        self_all_probabilities_view = self.all_probabilities[idx]
        #self_semantics_view = self.all_semantics[idx]
        self_c2w = self.all_poses[idx]
        self_intrinsics = self.all_intrinsics[idx]
        
        num_corres = self.all_mask_valid_corr[idx].shape[0]
        max_index = 0
        max_sum = self.all_mask_valid_corr[idx][0].sum()
        for i in range(1, num_corres):
            current_sum = self.all_mask_valid_corr[idx][i].sum()  
            if current_sum > max_sum:  
                max_index = i 
                max_sum = current_sum
        #print('max_index:',max_index)
        
        max_mask_corr = self.all_mask_valid_corr[idx][max_index].permute(1, 2, 0)  # (H, W, 1)
        max_mask_corr = max_mask_corr.squeeze(-1)  # (H, W)
        pixels_in_self_y = self.grid[max_mask_corr]  # [N_ray, 2], absolute pixel locations
        ray_in_self_int = self.grid_flat[max_mask_corr]# [N_ray]
        
        if ray_in_self_int.shape[0] > self.max_rays:
            random_values = torch.randperm(ray_in_self_int.shape[0])[:self.max_rays]
            #pixels_in_self = pixels_in_self[random_values]
            ray_in_self_int = ray_in_self_int[random_values]
        
        #self_rgbs = self_rgbs_view[ray_in_self_int]
        self_rays = self_rays_view[ray_in_self_int]
        self_confidences = self_confidences_view[ray_in_self_int]
        self_probabilities = self_all_probabilities_view[ray_in_self_int]
        #self_semantics = self_semantics_view[ray_in_self_int]
        pixels_in_self = self.grid.view(self.image_dim[0] * self.image_dim[1], -1)
        pixels_in_self = pixels_in_self[ray_in_self_int]
        
        all_other_c2w = []
        all_other_intrinsics = []
        all_pixels_in_other = []
        all_conf_corrr = []
        all_mask_corr = []
        all_other_rays = []
        all_other_confidences = []
        all_other_probabilities = []
        corres_idxs_view = self.all_corres_idx[idx]
        for i, corres_idx in enumerate(corres_idxs_view):
            #other_rgbs_view = self.all_rgbs[idx+corres_idx]
            other_rays_view = self.all_rays[idx+corres_idx]
            other_confidences_view = self.all_confidences[idx+corres_idx]
            other_probabilities_view = self.all_probabilities[idx+corres_idx]
            #other_semantics_view = self.all_semantics[idx+corres_idx]
            other_c2w = self.all_poses[idx+corres_idx]
            other_intrinsics = self.all_intrinsics[idx+corres_idx]
            
            conf_map_self_to_other = self.all_conf_maps[idx][i].permute(1, 2, 0)  # (H, W, 1)
            conf_map_self_to_other = conf_map_self_to_other.view(self.image_dim[0] * self.image_dim[1], 1)
            mask_correct_corr = self.all_mask_valid_corr[idx][i].permute(1, 2, 0)  # (H, W, 1)
            mask_correct_corr = mask_correct_corr.view(self.image_dim[0] * self.image_dim[1], 1)
            
            corres_map_self_to_other = self.all_corres_maps[idx][i].permute(1, 2, 0) # (H, W, 2)
            corres_map_self_to_other = corres_map_self_to_other.view(self.image_dim[0] * self.image_dim[1], -1)
            pixels_in_other = corres_map_self_to_other[ray_in_self_int] # [N_ray, 2], absolute pixel locations
            conf_corrr = conf_map_self_to_other[ray_in_self_int]
            mask_corr = mask_correct_corr[ray_in_self_int]
            
            corres_map_self_to_other_rounded = torch.round(corres_map_self_to_other).long()
            corres_map_self_to_other_rounded_flat = corres_map_self_to_other_rounded[:, 1] * self.image_dim[1] + corres_map_self_to_other_rounded[:, 0]
            ray_in_other_int = corres_map_self_to_other_rounded_flat[ray_in_self_int]
            ray_in_other_int_clip = torch.where(ray_in_other_int > 65535, 65535, ray_in_other_int)
            ray_in_other_int_clip = torch.where(ray_in_other_int_clip < 0, 0, ray_in_other_int_clip)
            
            #other_rgbs = other_rgbs_view[ray_in_other_int_clip]
            other_rays = other_rays_view[ray_in_other_int_clip]
            other_confidences = other_confidences_view[ray_in_other_int_clip]
            other_probabilities = other_probabilities_view[ray_in_other_int_clip]
            #other_semantics = other_semantics_view[ray_in_other_int_clip]
            
            all_other_c2w.append(other_c2w)
            all_other_intrinsics.append(other_intrinsics)
            all_pixels_in_other.append(pixels_in_other)
            all_conf_corrr.append(conf_corrr)
            all_mask_corr.append(mask_corr)
            all_other_rays.append(other_rays)
            all_other_confidences.append(other_confidences)
            all_other_probabilities.append(other_probabilities)
            
        sample = {
            f"self_rays": self_rays,
            f"all_other_rays": all_other_rays,
            f"self_confidences": self_confidences,
            f"all_other_confidences": all_other_confidences,
            f"self_probabilities": self_probabilities,
            f"all_other_probabilities": all_other_probabilities,
            f"self_c2w": self_c2w,
            f"all_other_c2w": all_other_c2w,
            f"self_intrinsics": self_intrinsics,
            f"all_other_intrinsics": all_other_intrinsics,
            f"pixels_in_self": pixels_in_self,
            f"all_pixels_in_other": all_pixels_in_other,
            f"all_conf_corrr": all_conf_corrr,
            f"all_mask_corr": all_mask_corr,
            f"max_index": max_index,
        }
        return sample

    def __len__(self):
        #print('all_corres_maps:',len(self.all_corres_maps))
        return len(self.all_corres_maps)

    @staticmethod
    def collate_fn(batch):
        return {
            "self_rays": [x["self_rays"] for x in batch],
            "all_other_rays": [x["all_other_rays"] for x in batch],
            "self_confidences": [x["self_confidences"] for x in batch],
            "all_other_confidences": [x["all_other_confidences"] for x in batch],
            "self_probabilities": [x["self_probabilities"] for x in batch],
            "all_other_probabilities": [x["all_other_probabilities"] for x in batch],
            "self_c2w": [x["self_c2w"] for x in batch],
            "all_other_c2w": [x["all_other_c2w"] for x in batch],
            "self_intrinsics": [x["self_intrinsics"] for x in batch],
            "all_other_intrinsics": [x["all_other_intrinsics"] for x in batch],
            "pixels_in_self": [x["pixels_in_self"] for x in batch],
            "all_pixels_in_other": [x["all_pixels_in_other"] for x in batch],
            "all_conf_corrr": [x["all_conf_corrr"] for x in batch],
            "all_mask_corr": [x["all_mask_corr"] for x in batch],
            "max_index": [x["max_index"] for x in batch],
        }
