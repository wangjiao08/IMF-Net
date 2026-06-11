import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=70000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--home', type=str, default='/home/wangjiao/Heavy_precipitation')

    args = parser.parse_args()
    args.hw = 64
    args.input_length = args.in_len = 2
    args.output_length = args.out_len = 2
    args.in_chans = 12
    args.out_chans = 1
    args.depths = [2, 6, 2, 2]
    args.frozen_stages = None  #当前不冻结任何网络层，整网都参与训练
    args.upsampling_scale = (1,2,2)
    args.patch_expan_size = (2,4,4)


    stats_file = f'{args.home}/64_4km/normalization_64_npy/normalization_stats.txt'
    stats = {}
    with open(stats_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(': ')
                stats[key] = value
    name = 'pre'
    args.channel_min = float(stats[f'{name}_min'])
    args.channel_max = float(stats[f'{name}_max'])

    args.batch_size = 32
    args.thresholds = [0.1, 1, 5, 10, 20, 25, 30, 50]
    args.dataloader_thread = 0

    return args