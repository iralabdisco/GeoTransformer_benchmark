import argparse

import torch
import numpy as np
import open3d as o3d

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model
from dataset import test_data_loader


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):

    src_pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(args.src_file)
    ref_pcd: o3d.geometry.PointCloud  = o3d.io.read_point_cloud(args.ref_file)

    src_pcd = src_pcd.voxel_down_sample(0.3)
    ref_pcd = ref_pcd.voxel_down_sample(0.3)

    src_points = np.asarray(src_pcd.points)
    ref_points = np.asarray(ref_pcd.points)

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)

    #[76 65 75 76 73]
    _, neighbor_limits = test_data_loader(cfg)
    print(neighbor_limits)

    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    # prepare model

    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'], strict=True)

    # prediction
    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"]

    # compute error
    rre, rte = compute_registration_error(transform, estimated_transform)
    print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()