import argparse

import torch
import numpy as np
import pandas as pd
import os
import sys
import csv
from tqdm import tqdm
import copy
import open3d as o3d

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error
from dataset import test_data_loader

from config import make_cfg
from model import create_model

import benchmark_helpers

def load_data(src_pcd, ref_pcd):
    # src_pcd = src_pcd.voxel_down_sample(0.3)
    # ref_pcd = ref_pcd.voxel_down_sample(0.3)

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

    transform = np.random.rand(4,4)
    data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')

    # Benchmark files and dirs
    parser.add_argument('--input_txt', type=str,
                        help='Path to the problem .txt')
    parser.add_argument('--input_pcd_dir', type=str,
                        help='Path to the pcd directory')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save results to')
    # Model arguments
    parser.add_argument('--weights', type=str,
                        help='Path to pretrained model')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Model setup
    cfg = make_cfg()
    _, neighbor_limits = test_data_loader(cfg) # TODO sembra non essere un qualcosa deterministico, anche se forse dovrebbe
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'], strict=True)
    model.eval()

    # Load problems txt file
    df = pd.read_csv(args.input_txt, sep=' ', comment='#')
    df = df.reset_index()
    problem_name = os.path.splitext(os.path.basename(args.input_txt))[0]

    # initialize result file
    os.makedirs(args.output_dir, exist_ok=True)
    header_comment = "# " + " ".join(sys.argv[:]) + "\n"
    header = ['id', 'initial_error', 'final_error', 'transformation', 'status_code']
    result_name = problem_name + "_result.txt"
    result_filename = os.path.join(args.output_dir, result_name)
    with open(result_filename, mode='w') as f:
        f.write(header_comment)
        csv_writer = csv.writer(f, delimiter=';')
        csv_writer.writerow(header)

    # Solve for each problem
    n_fails_oom = 0
    n_fails_other = 0
    print(problem_name)
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        problem_id, source_pcd, target_pcd, source_transform, target_pcd_filename = \
            benchmark_helpers.load_problem(row, args)

        # calculate initial error
        moved_source_pcd = copy.deepcopy(source_pcd)
        moved_source_pcd.transform(source_transform)
        initial_error = benchmark_helpers.calculate_error(source_pcd, moved_source_pcd)

        data_dict = load_data(moved_source_pcd, target_pcd)
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius,
            neighbor_limits
        )

        # prediction

        data_dict = to_cuda(data_dict)
        try:
            output_dict = model(data_dict)
            output_dict = release_cuda(output_dict)
            data_dict = release_cuda(data_dict)
            registration_solution = output_dict["estimated_transform"]
            error = "ok"
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory.'):
                error = "OOM"
                n_fails_oom += 1
            else:
                error = "runtime_error"
                n_fails_other += 1
            data_dict = release_cuda(data_dict)
            registration_solution = np.eye(4)

        # calculate final error
        moved_source_pcd.transform(registration_solution)
        final_error = benchmark_helpers.calculate_error(source_pcd, moved_source_pcd)

        # write results to file
        str_solution = ' '.join(map(str, registration_solution.ravel()))
        results = [problem_id, initial_error, final_error,
                   str_solution, error]
        with open(result_filename, mode='a') as f:
            csv_writer = csv.writer(
                f, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
            csv_writer.writerow(results)

    print("Number of failures oom: ", n_fails_oom)
    print("Number of failures other: ", n_fails_other)
    exit()


if __name__ == '__main__':
    main()