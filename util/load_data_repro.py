import random
import netCDF4 as nc
import xarray as xr
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from functools import lru_cache
from collections import defaultdict


def load_cache_data(cfg):
    total_len = cfg.in_len + cfg.out_len
    seed = cfg.seed

    def load_grid_labels(label_path):
        """加载.npy标签文件，返回格式：[(文件名, 网格索引, 标签), ...]"""
        labels = np.load(label_path, allow_pickle=True)
        return labels

    def get_balanced_train_samples(labels, nc_files, valid_sequences, blocks, required_length=total_len,
                                   pos_neg_ratio=(9, 1)):
        """
        优化版：通过字典索引加速样本筛选，减少重复计算
        """
        # 步骤1：预处理labels为字典，key=(文件名, 网格索引)，value=标签
        label_dict = defaultdict(str)
        for row in labels:
            # 假设labels的每行格式为 [文件名, 网格索引字符串, 标签]
            filename, block_idx, label = row
            label_dict[(filename, block_idx)] = label

        pos_samples = []  # 正样本：(文件路径, 网格索引)
        neg_samples = []  # 负样本：(文件路径, 网格索引)
        # 【修改点1】：预计算所有文件的basename映射
        # 关键：因为现在读的是 .npy 文件，但 label 里的文件名是 .nc
        # 所以这里强制把 .npy 替换回 .nc，以确保能匹配上标签字典
        nc_basename = {path: os.path.basename(path).replace('.npy', '.nc') for path in nc_files}

        for i in valid_sequences:
            for block in range(blocks):
                block_str = str(block)  # 转为字符串，与labels中的格式一致

                is_neg = True  # 只要有一个不是负样本(只要有一个正)，就标记为正样本，以前64_4km用的这个
                # 检查连续required_length个序列是否全为负样本
                for j in range(i, i + required_length):   #原本是 i + required_length-1
                    # 直接从预计算的字典获取文件名 (.nc后缀)
                    target_file = nc_basename[nc_files[j]]
                    # 从字典查询标签，O(1)复杂度
                    current_label = label_dict.get((target_file, block_str), '0')

                    if current_label != '0':
                        is_neg = False
                        break
                if is_neg:
                    neg_samples.append((nc_files[i], block))
                else:
                    pos_samples.append((nc_files[i], block))



        pos_count = len(pos_samples)
        neg_count = len(neg_samples)
        print(f"原始样本 正样本: {pos_count}, 负样本: {neg_count}")

        # 步骤2：平衡正负样本（与原逻辑保持一致）
        if pos_count == 0 or neg_count == 0:
            raise ValueError("训练集中正负样本至少有一类为0，无法平衡！")

        pos_ratio, neg_ratio = pos_neg_ratio

        total_target = 102776  # 要和验证集成8：2        # print((102776 + 25694))
        # total_target = 1000000000000000000  #全部都用



            # 计算目标正负样本数
        target_pos = int(total_target * pos_ratio / 10.0)
        target_neg = total_target - target_pos

        # 检查原始样本是否足够
        if target_pos > pos_count:
            print(f"警告: 正样本不足，需要{target_pos}但只有{pos_count}，使用全部正样本")
            target_pos = pos_count
            # 重新计算负样本数以保持比例
            target_neg = min(neg_count, int(target_pos * neg_ratio / pos_ratio))

        if target_neg > neg_count:
            print(f"警告: 负样本不足，需要{target_neg}但只有{neg_count}，使用全部负样本")
            target_neg = neg_count
            # 重新计算正样本数以保持比例
            target_pos = min(pos_count, int(target_neg * pos_ratio / neg_ratio))

        # 随机下采样
        random.seed(seed)
        balanced_pos = random.sample(pos_samples, target_pos) if target_pos > 0 else []
        balanced_neg = random.sample(neg_samples, target_neg) if target_neg > 0 else []

        # 合并并打乱
        balanced_samples = balanced_pos + balanced_neg
        random.shuffle(balanced_samples)

        print(f"平衡后 正样本:{len(balanced_pos)}, 负样本:{len(balanced_neg)}")
        print(
            f"最终比例: 正{len(balanced_pos) / len(balanced_samples):.1%}, 负{len(balanced_neg) / len(balanced_samples):.1%}")

        return balanced_samples

    def get_time_ordered_files(data_folder):
        """获取按时间排序的文件列表"""
        # 【修改点2】：只读取 .npy 文件
        nc_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]
        # 按文件名中的时间排序
        nc_files.sort(key=lambda x: parse_filename(os.path.basename(x)))
        return nc_files

    def parse_filename(filename):
        """解析文件名中的时间（针对YYYYMMDDHH.npy格式）"""
        try:
            # 移除扩展名，得到纯数字时间字符串
            time_str = os.path.splitext(filename)[0]
            # 直接使用10位数字格式（YYYYMMDDHH）
            return datetime.strptime(time_str, '%Y%m%d%H')
            # return datetime.strptime(time_str, '%Y%m%d%H%M')

        except Exception as e:
            print(f"解析文件名 {filename} 时出错: {e}")
            raise
    def find_continuous_sequences(nc_files, required_length=total_len, interval=timedelta(hours=1)):
        """
        查找所有连续的时间序列
        """
        if len(nc_files) < required_length:
            return []

        # 获取所有文件的时间戳
        file_times = [parse_filename(os.path.basename(f)) for f in nc_files]
        continuous_starts = []

        # 滑动窗口检查连续序列
        for i in range(len(nc_files) - required_length + 1):
            # 检查从i开始的required_length个文件是否连续
            is_continuous = True
            for j in range(i, i + required_length - 1):
                if file_times[j + 1] - file_times[j] != interval:
                    is_continuous = False
                    break
            if is_continuous:
                continuous_starts.append(i)  # 记录连续序列的起始索引

        print(f"找到 {len(continuous_starts)} 个连续的 {required_length} 序列")
        return continuous_starts

    class BalancedDataset(Dataset):
        def __init__(self, data_folder, flag, label_path=None, use_aug=True, base_seed=42):
            self.data_folder = data_folder
            self.flag = flag
            self.label_path = label_path
            self.use_aug = use_aug
            self.base_seed = base_seed

            # 步骤1：获取按时间排序的文件列表 (.npy)
            self.nc_files = get_time_ordered_files(data_folder)
            self.blocks_per_file = self._get_blocks_per_file(self.nc_files)
            self.total_files = len(self.nc_files)

            # 找到所有连续的4小时序列
            self.valid_sequences = find_continuous_sequences(self.nc_files, required_length=total_len)
            if not self.valid_sequences and self.flag != 'train':
                raise ValueError(f"[{self.flag}] 没有找到足够的连续{total_len}小时序列，请检查数据！")

            # 步骤2：样本生成逻辑
            if (self.flag == 'train' or self.flag == 'val') and self.label_path is not None:
                self.labels = load_grid_labels(label_path)
                self.balanced_samples = get_balanced_train_samples(self.labels, self.nc_files, self.valid_sequences, self.blocks_per_file)
                self.total_samples = len(self.balanced_samples)
            else:
                # 只使用有效的连续序列
                self.samples = self.valid_sequences
                self.total_samples = len(self.samples) * self.blocks_per_file

            print(f"[{self.flag}] 数据集初始化完成，总样本数：{self.total_samples}")
            if self.total_samples == 0:
                print(f"警告: [{self.flag}] 数据集样本数为0，请检查数据！")

        def __len__(self):
            return self.total_samples

        def _get_blocks_per_file(self, nc_files):
            # 【修改点3】：读取 .npy 的 shape，使用 mmap 模式极其快速
            # 不再需要 xarray 或 netCDF4
            return np.load(nc_files[0], mmap_mode='r').shape[0]

        # 【修改点4】：移除 @lru_cache
        # 原因：mmap 本身就是操作系统级缓存，加 lru_cache 会导致文件句柄不释放，且没必要
        def _load_cached_file(self, file_path):
            """
            核心修改：使用 mmap 读取 npy 文件
            return: memmap 对象 (类似 numpy 数组，但数据在硬盘上)
            """
            # mmap_mode='r'：只读模式，不占物理内存，直到被索引切片
            return np.load(file_path, mmap_mode='r')

        def _get_file_data(self, file_idx):
            """获取文件数据"""
            file_path = self.nc_files[file_idx]
            return self._load_cached_file(file_path)

        def __getitem__(self, idx):
            if (self.flag == 'train' or self.flag == 'val') and self.label_path is not None:
                nc_file_path, grid_idx = self.balanced_samples[idx]
                seq_start = self.nc_files.index(nc_file_path)
            else:
                sample_idx = idx // self.blocks_per_file
                grid_idx = idx % self.blocks_per_file
                seq_start = self.samples[sample_idx]  # 使用预筛选的有效序列

            # 获取连续的4个时间步
            frames = []
            for i in range(total_len):
                file_idx = seq_start + i

                # 这里获取的是 mmap 对象，几乎不耗时
                file_data = self._get_file_data(file_idx)

                # 【关键】：file_data[grid_idx] 这一步才会真正触发 IO
                # 操作系统只读取这一小块数据到内存
                frames.append(file_data[grid_idx])

            frames = np.array(frames)
            #
            # 【新增】数据增强 (仅针对训练集)
            # ==========================================
            if self.flag == 'train' and self.use_aug:
                # 让增强只依赖样本idx和基础seed，而不依赖外部全局随机状态
                rng = random.Random(self.base_seed + idx)

                # 1. 随机水平翻转
                if rng.random() > 0.5:
                    # frames是numpy数组，在最后两个维度(H, W)上操作
                    frames = np.flip(frames, axis=-1)

                # 2. 随机垂直翻转
                if rng.random() > 0.5:
                    frames = np.flip(frames, axis=-2)

                # 3. 随机旋转 (0, 90, 180, 270度)
                k = rng.choice([0, 1, 2, 3])
                if k > 0:
                    # axes=(-2, -1) 表示在最后两个维度(H, W)上旋转
                    frames = np.rot90(frames, k=k, axes=(-2, -1))

            return frames.copy()  # t,c,h,w

    data_folder = f"{cfg.home}/64_4km/normalization_64_npy"


    data_folders = {
        'train': os.path.join(data_folder, 'train'),
        'val': os.path.join(data_folder, 'val'),
        'test': os.path.join(data_folder, 'test_new')
    }

    binary_path_train = f'{cfg.home}/64_4km/binary_64/binary_train.npy'
    binary_path_val = f'{cfg.home}/64_4km/binary_64/binary_val.npy'


    # 初始化数据集
    train_dataset = BalancedDataset(
        data_folders['train'], 'train', binary_path_train,
        use_aug=True, base_seed=cfg.seed
    )
    val_dataset = BalancedDataset(
        data_folders['val'], 'val', binary_path_val,
        use_aug=False, base_seed=cfg.seed
    )
    test_dataset = BalancedDataset(
        data_folders['test'], 'test',
        use_aug=False, base_seed=cfg.seed
    )

    return train_dataset, val_dataset, test_dataset



