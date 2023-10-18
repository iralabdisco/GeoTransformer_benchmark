import argparse

import torch
import numpy as np
import open3d as o3d

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model


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
    parser.add_argument('--model_path', type=str,
                        help='Path to pretrained model')
    parser.add_argument('--n_iter', type=int,
                        help='Number of iteration for registration')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Pytorch setup and load model
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("No GPU available")

    fmr = Demo(args.n_iter)
    model = fmr.create_model()
    pretrained_path = args.model_path
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.to(device)

    # Load problems txt file
    df = pd.read_csv(args.input_txt, sep=' ', comment='#')
    df = df.reset_index()
    problem_name = os.path.splitext(os.path.basename(args.input_txt))[0]

    # initialize result file
    os.makedirs(args.output_dir, exist_ok=True)
    header_comment = "# " + " ".join(sys.argv[:]) + "\n"
    header = ['id', 'initial_error', 'final_error', 'transformation']
    result_name = problem_name + "_result.txt"
    result_filename = os.path.join(args.output_dir, result_name)
    with open(result_filename, mode='w') as f:
        f.write(header_comment)
        csv_writer = csv.writer(f, delimiter=';')
        csv_writer.writerow(header)

    # Solve for each problem
    print(problem_name)
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        problem_id, source_pcd, target_pcd, source_transform, target_pcd_filename = \
            benchmark_helper.load_problem(row, args)

        # calculate initial error
        moved_source_pcd = copy.deepcopy(source_pcd)
        moved_source_pcd.transform(source_transform)
        initial_error = metric.calculate_error(source_pcd, moved_source_pcd)

        # downsample and convert to np
        down_src = moved_source_pcd.voxel_down_sample(voxel_size=args.voxel_size)
        src_p = np.asarray(down_src.points)
        src_p = np.expand_dims(src_p, 0)

        down_target = target_pcd.voxel_down_sample(voxel_size=args.voxel_size)
        target_p = np.asarray(down_target.points)
        target_p = np.expand_dims(target_p, 0)

        # solve
        registration_solution = fmr.evaluate(model, target_p, src_p, device)

        registration_solution = registration_solution.detach().cpu().numpy()

        # calculate final error
        moved_source_pcd.transform(registration_solution)
        final_error = metric.calculate_error(source_pcd, moved_source_pcd)

        # write results to file
        str_solution = ' '.join(map(str, registration_solution.ravel()))
        results = [problem_id, initial_error, final_error,
                   str_solution]
        with open(result_filename, mode='a') as f:
            csv_writer = csv.writer(
                f, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
            csv_writer.writerow(results)


if __name__ == '__main__':
    main()