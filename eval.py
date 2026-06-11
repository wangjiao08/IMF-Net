from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler # Add this import
from util.metrics import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os



def calculate_full_metrics(pred_full,pred_unet_full, true_full,cfg,return_stats):
    thresholds = cfg.thresholds
    """
    在完整的大图上计算指标
    pred_full: Fuxi回归预测 (Tensor, H, W)
    true_full: 真实值 (Tensor, H, W)
    pred_unet_full: U-Net分类预测 (Tensor, H, W, 值为0-8)
    thresholds: 阈值列表
    """
    # 1. 统一转为 Tensor，并放在 CPU 上算指标，避免评估阶段爆显存
    if not isinstance(pred_full, torch.Tensor):
        pred_full = torch.from_numpy(pred_full)
    if not isinstance(true_full, torch.Tensor):
        true_full = torch.from_numpy(true_full)
    if not isinstance(pred_unet_full, torch.Tensor):
        pred_unet_full = torch.from_numpy(pred_unet_full)

    pred_full = pred_full.detach().cpu().float()
    true_full = true_full.detach().cpu().float()
    pred_unet_full = pred_unet_full.detach().cpu().long()

    # 2. 生成分类真值
    thresholds_tensor = torch.tensor(thresholds, device='cpu', dtype=true_full.dtype)
    target_classes = torch.bucketize(true_full, thresholds_tensor, right=True)

    # 3. 计算基础指标，直接用 CPU tensor 计算
    batch_mse = torch.mean((true_full - pred_full) ** 2).item()
    batch_mae = torch.mean(torch.abs(true_full - pred_full)).item()


    print(f"  MSE : {batch_mse:.4f}")
    print(f"  MAE : {batch_mae:.4f}")

    # 用于返回的统计字典
    stats_dict = {
        'mse': batch_mse,
        'mae': batch_mae,
        'count': pred_full.shape[0],  # 当前批次的样本数
        'thresholds': {}
    }

    # 4. 计算分级指标
    eps = 1e-6
    for i, t in enumerate(thresholds):
        print(f"\n>>> 降水阈值 >= {t} mm/h")

        # 记录该阈值下的混淆矩阵
        stats_dict['thresholds'][t] = {}

        # === U-Net 分类表现 ===
        # 逻辑：第 i 个阈值对应 Class i+1
        class_cutoff = i + 1
        u_true_bin = target_classes >= class_cutoff
        u_pred_bin = pred_unet_full >= class_cutoff

        u_tp = torch.sum(u_true_bin & u_pred_bin).item()
        u_fn = torch.sum(u_true_bin & (~u_pred_bin)).item()
        u_fp = torch.sum((~u_true_bin) & u_pred_bin).item()
        u_tn = torch.sum((~u_true_bin) & (~u_pred_bin)).item()

        # 存入字典
        stats_dict['thresholds'][t]['unet'] = {'tp': u_tp, 'fn': u_fn, 'fp': u_fp, 'tn': u_tn}

        u_acc = (u_tp + u_tn) / (u_tp + u_fn + u_fp + u_tn + eps)
        u_precision = u_tp / (u_tp + u_fp + eps)
        u_recall = u_tp / (u_tp + u_fn + eps)
        u_f1 = 2 * (u_precision * u_recall) / (u_precision + u_recall + eps)
        u_po = u_fn / (u_tp + u_fn + eps)
        u_far = u_fp / (u_tp + u_fp + eps)
        u_csi = u_tp / (u_tp + u_fn + u_fp + eps)

        print(f"  [U-Net 分类表现]")
        print(f"    Accuracy : {u_acc:.4f}")
        print(f"    Precision: {u_precision:.4f}")
        print(f"    Recall   : {u_recall:.4f}")
        print(f"    F1 Score : {u_f1:.4f}")
        print(f"    PO (Miss): {u_po:.4f}")
        print(f"    FAR      : {u_far:.4f}")
        print(f"    CSI      : {u_csi:.4f}")

        # === Fuxi 回归表现 ===
        r_true_bin = true_full >= t
        r_pred_bin = pred_full >= t

        r_tp = torch.sum(r_true_bin & r_pred_bin).item()
        r_fn = torch.sum(r_true_bin & (~r_pred_bin)).item()
        r_fp = torch.sum((~r_true_bin) & r_pred_bin).item()
        r_tn = torch.sum((~r_true_bin) & (~r_pred_bin)).item()


        # 存入字典
        stats_dict['thresholds'][t]['fuxi'] = {'tp': r_tp, 'fn': r_fn, 'fp': r_fp, 'tn': r_tn}

        r_csi = r_tp / (r_tp + r_fn + r_fp + eps)
        r_hss = (2 * (r_tp * r_tn - r_fn * r_fp)) / (
                    (r_tp + r_fn) * (r_fn + r_tn) + (r_tp + r_fp) * (r_fp + r_tn) + eps)
        r_far = r_fp / (r_tp + r_fp + eps)
        r_bias = (r_tp + r_fp) / (r_tp + r_fn + eps)
        r_po = r_fn / (r_tp + r_fn + eps)

        print(f"  [Fuxi 回归表现]")
        print(f"    CSI      : {r_csi:.4f}")
        print(f"    HSS      : {r_hss:.4f}")
        print(f"    FAR      : {r_far:.4f}")
        print(f"    BIAS     : {r_bias:.4f}")
        print(f"    PO       : {r_po:.4f}")

    if return_stats:
        return stats_dict


# ========================= 新增：短时强降水业务评分（4km/1h逐小时） =========================
TABLE4_BIN_EDGES = np.array([50.0, 60.0, 70.0, 80.0, 90.0, 100.0], dtype=np.float32)
TABLE4_SCORE = np.array([
    [np.nan, 0, 0, 0, 0, 0, 0],
    [0, 3, 2, 1, 0, 0, 0],
    [0, 2, 3, 2, 1, 0, 0],
    [0, 1, 2, 3, 2, 1, 0],
    [0, 0, 1, 2, 3, 2, 1],
    [0, 0, 0, 1, 2, 3, 2],
    [0, 0, 0, 0, 1, 2, 3],
], dtype=np.float32)


def strong_point_to_area_counts(pred, true, threshold=20.0, radius_px=5):
    """
    pred, true: [B, H, W] 或 [H, W]，单位 mm/h
    返回：NA, NB, NC, ND
    4km分辨率下，20km扫描半径约等于5个像素。
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if true.ndim == 2:
        true = true.unsqueeze(0)

    pred = pred.float()
    true = true.float()

    pred_bin = (pred >= threshold).float().unsqueeze(1)
    true_bin = (true >= threshold).float().unsqueeze(1)

    # 4km数据用20km扫描半径 => 5个像素，使用max_pool2d避免大卷积开销
    kernel_size = 2 * radius_px + 1
    pred_expand = F.max_pool2d(pred_bin, kernel_size=kernel_size, stride=1, padding=radius_px) > 0
    true_expand = F.max_pool2d(true_bin, kernel_size=kernel_size, stride=1, padding=radius_px) > 0

    pred_event = pred_bin > 0
    true_event = true_bin > 0

    # 以观测为主统计命中/漏报，以预报为主统计空报
    NA = (true_event & pred_expand).sum().item()
    NB = (pred_event & (~true_expand)).sum().item()
    NC = (true_event & (~pred_expand)).sum().item()

    total = pred.shape[0] * pred.shape[1] * pred.shape[2]
    ND = total - NA - NB - NC
    if ND < 0:
        ND = 0

    return NA, NB, NC, ND


def strong_ts_bias_from_counts(NA, NB, NC):
    """按图1中的特殊情况约定计算 TS / BIAS"""
    if (NA + NB + NC) == 0:
        TS = 999999
    else:
        TS = NA / (NA + NB + NC)

    if (NA + NC) == 0 and (NA + NB) == 0:
        BIAS = 0
    elif (NA + NC) == 0 and (NA + NB) > 0:
        BIAS = 999999
    else:
        BIAS = abs((NA + NB) / (NA + NC) - 1)

    return TS, BIAS


def strong_amount_me_score(pred_maps, true_maps, radius_px=5, obs_threshold=50.0,
                           bin_edges=None, score_table=None):
    """
    按图2表4做“强短时强降水降水量相对误差得分”。
    这里采用点对面思想：对每个观测强降水格点，在预报半径20km邻域内取“最优得分”。
    pred_maps, true_maps: [B, H, W] 或 [H, W]，单位 mm/h
    返回：{'sum': 总得分, 'count': 参与评分格点数, 'mean': 平均得分}
    """
    if isinstance(pred_maps, torch.Tensor):
        pred_maps = pred_maps.detach().cpu().numpy()
    if isinstance(true_maps, torch.Tensor):
        true_maps = true_maps.detach().cpu().numpy()

    pred_maps = np.asarray(pred_maps, dtype=np.float32)
    true_maps = np.asarray(true_maps, dtype=np.float32)

    if pred_maps.ndim == 2:
        pred_maps = pred_maps[None, ...]
    if true_maps.ndim == 2:
        true_maps = true_maps[None, ...]

    if bin_edges is None:
        bin_edges = TABLE4_BIN_EDGES
    if score_table is None:
        score_table = TABLE4_SCORE

    r = int(radius_px)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= (r * r)

    pred_cls = np.digitize(pred_maps, bin_edges, right=False)
    true_cls = np.digitize(true_maps, bin_edges, right=False)

    total_score = 0.0
    total_count = 0

    B, H, W = true_maps.shape
    for b in range(B):
        coords = np.argwhere(true_maps[b] >= obs_threshold)
        for y, x in coords:
            obs_cls = true_cls[b, y, x]

            y0 = max(0, y - r)
            y1 = min(H, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(W, x + r + 1)

            ky0 = r - (y - y0)
            ky1 = ky0 + (y1 - y0)
            kx0 = r - (x - x0)
            kx1 = kx0 + (x1 - x0)

            disk_view = disk[ky0:ky1, kx0:kx1]
            neigh_pred_cls = pred_cls[b, y0:y1, x0:x1][disk_view]

            scores = score_table[neigh_pred_cls, obs_cls]
            best_score = np.nanmax(scores)

            total_score += float(best_score)
            total_count += 1

    mean_score = np.nan if total_count == 0 else (total_score / total_count)
    return {'sum': total_score, 'count': total_count, 'mean': mean_score}

def evaluation(cfg, test_loader,model):
    from cropped_64_DATA.Restore_crop import Restore
    channel_min = cfg.channel_min
    channel_max = cfg.channel_max
    # 1. 加载模型
    model.load_state_dict(torch.load(cfg.model_save_best))
    flag  = cfg.model_save_best.split('/')[-1].split('.')[0]

    model.eval()

    # 临时列表，用于存储每个 Batch 的结果
    preds_list = []
    unets_list = []
    truths_list = []

    print(">>> 正在进行模型推理 (Inference)...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(cfg.device)  # [B, T, C, H, W]

            # 准备输入和真值

            #全部12个变量
            x = batch[:, :cfg.in_len, :, :, :].permute(0, 2, 1, 3, 4)
            truth = batch[:, -cfg.out_len:, 3:4, :, :].permute(0, 2, 1, 3, 4)

            # 推理
            pred, out_unet,_= model(x)  #(b,c,t,h,w)=(b,1,2,64,64) (b,9,2,64,64)
            # pred, out_unet= model(x)  #(b,c,t,h,w)=(b,1,2,64,64) (b,9,2,64,64)

            # 收集数据 (保持 T 在第0维，方便后面按时间切分)
            # pred[:, 0] shape: (B, T, 64, 64) -> permute -> (T, B, 64, 64)
            preds_list.append(pred[:, 0].permute(1, 0, 2, 3).cpu())

            unets_list.append(torch.argmax(out_unet, dim=1).permute(1, 0, 2, 3).cpu())

            truths_list.append(truth[:, 0].permute(1, 0, 2, 3).cpu())

    # 2. 将所有 Batch 拼接成巨大的 Tensor
    # 拼接第1维 (Batch维)，结果 shape: (T, Total_Tiles, H, W)
    # Total_Tiles 是测试集所有切片的总数
    all_preds = torch.cat(preds_list, dim=1)
    all_unets = torch.cat(unets_list, dim=1)
    all_truths = torch.cat(truths_list, dim=1)

    # 获取参数
    T, Total_Tiles, H, W = all_preds.shape
    # 设定每张大图由多少个切片组成 (这个数必须准确！)
    # 假设你的测试集是按顺序排列的，且每张图由 blocks_per_file 个切片组成
    # 这里你需要从 dataset 获取
    num_tiles_per_image = test_loader.dataset.blocks_per_file

    # --- 初始化全局累加器 ---
    global_accum = {
        'total_mse': 0.0,
        'total_mae': 0.0,
        'total_count': 0,
        'thresholds': {t: {'unet': {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0},
                           'fuxi': {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}} for t in cfg.thresholds}
    }

    # 新增：短时强降水业务评分累加器（4km/1h不需要再聚合）
    def init_strong_accum():
        return {'NA': 0.0, 'NB': 0.0, 'NC': 0.0, 'ND': 0.0, 'ME_SUM': 0.0, 'ME_COUNT': 0}

    strong_global_accum = init_strong_accum()
    strong_step_accums = [init_strong_accum() for _ in range(cfg.out_len)]

    strong_ts_threshold = getattr(cfg, 'strong_ts_threshold', 20.0)
    strong_me_obs_threshold = getattr(cfg, 'strong_me_obs_threshold', 50.0)
    # 4km分辨率下20km扫描半径≈5个像素
    strong_radius_px = getattr(cfg, 'strong_radius_px', 5)

    # 3. 按时间步循环
    for t in range(cfg.out_len):
        print(f"\n>>> 正在评估第 {t + 1} 个预测时刻...")

        full_preds_t = []
        full_unets_t = []
        full_truths_t = []

        # 4. 按图片循环 (每次处理 num_tiles_per_image 个切片来还原一张图)
        for i in tqdm(range(0, Total_Tiles, num_tiles_per_image), desc=f"还原时刻 {t + 1} 的大图"):
            # 取出属于同一张大图的所有切片
            # shape: (num_tiles_per_image, 64, 64)
            curr_pred_tiles = all_preds[t, i: i + num_tiles_per_image]
            curr_unet_tiles = all_unets[t, i: i + num_tiles_per_image]
            curr_truth_tiles = all_truths[t, i: i + num_tiles_per_image]

            # 如果剩下一组不足以拼成一张图（可能是数据加载截断），跳过
            if curr_pred_tiles.shape[0] != num_tiles_per_image:
                continue

            # 反归一化
            denorm_pred = curr_pred_tiles * (channel_max - channel_min) + channel_min
            denorm_truth = curr_truth_tiles * (channel_max - channel_min) + channel_min
            # Unet 结果是类别索引，不需要反归一化，但 Restore 可能需要 float 类型

            # 还原拼接 (Restore 函数返回 numpy array (442, 631))
            # 注意：Restore 内部可能有保存文件的逻辑，如果不需要保存大量文件，建议修改 Restore 这里的逻辑
            # 这里传入 dummy filename 只是为了满足 Restore 接口，实际我们需要的是返回值
            model_name = cfg.model_save_best.split('/')[-1].split('.')[0]
            base_path = f'{cfg.home}/64_4km/Transunet_Swintransformer/result/{model_name}'
            os.makedirs(base_path, exist_ok=True)
            dummy_name = f"{base_path}/temp{flag}.nc"
            full_pred = Restore(denorm_pred.numpy(), dummy_name)
            full_unet = Restore(curr_unet_tiles.numpy(), dummy_name)
            full_truth = Restore(denorm_truth.numpy(), dummy_name)

            # 收集大图 (转回 Tensor 方便后续计算)
            full_preds_t.append(torch.from_numpy(full_pred))  #[(442,631),,,,,,,]
            full_unets_t.append(torch.from_numpy(full_unet))
            full_truths_t.append(torch.from_numpy(full_truth))


        # 注意：完整大图指标放在 CPU 上算，避免 GPU OOM
        full_preds_t = torch.stack(full_preds_t, dim=0).float()
        full_unets_t = torch.stack(full_unets_t, dim=0).long()
        full_truths_t = torch.stack(full_truths_t, dim=0).float()

        # 计算当前时刻的平均指标
        print(f"--- 第 {t + 1} 个时刻的平均指标 ---")
        stats = calculate_full_metrics(full_preds_t, full_unets_t, full_truths_t, cfg, return_stats=True)

        # ---------------------- 新增：短时强降水业务评分 ----------------------
        print(f"--- 第 {t + 1} 个时刻的短时强降水业务评分 ---")
        NA, NB, NC, ND = strong_point_to_area_counts(
            full_preds_t, full_truths_t,
            threshold=strong_ts_threshold,
            radius_px=strong_radius_px
        )
        TS, BIAS = strong_ts_bias_from_counts(NA, NB, NC)
        me_info = strong_amount_me_score(
            full_preds_t, full_truths_t,
            radius_px=strong_radius_px,
            obs_threshold=strong_me_obs_threshold
        )

        print(f"  强降水判识阈值 T : {strong_ts_threshold} mm/h")
        print(f"  扫描半径         : 20 km (~{strong_radius_px} px @ 4km)")
        print(f"  TS              : {TS}")
        print(f"  BIAS            : {BIAS}")
        if me_info['count'] == 0:
            print(f"  ME_SCORE        : nan (当前样本中无观测 >= {strong_me_obs_threshold} mm/h 的格点)")
        else:
            print(f"  ME_SCORE        : {me_info['mean']:.4f}")

        strong_step_accums[t]['NA'] += NA
        strong_step_accums[t]['NB'] += NB
        strong_step_accums[t]['NC'] += NC
        strong_step_accums[t]['ND'] += ND
        strong_step_accums[t]['ME_SUM'] += me_info['sum']
        strong_step_accums[t]['ME_COUNT'] += me_info['count']

        strong_global_accum['NA'] += NA
        strong_global_accum['NB'] += NB
        strong_global_accum['NC'] += NC
        strong_global_accum['ND'] += ND
        strong_global_accum['ME_SUM'] += me_info['sum']
        strong_global_accum['ME_COUNT'] += me_info['count']
        # -------------------------------------------------------------------------

        # --- 累加到全局累加器 ---
        count = stats['count']
        global_accum['total_count'] += count
        global_accum['total_mse'] += stats['mse'] * count  # 加权累加
        global_accum['total_mae'] += stats['mae'] * count

        for thresh in cfg.thresholds:
            for model_type in ['unet', 'fuxi']:
                for key in ['tp', 'fn', 'fp', 'tn']:
                    global_accum['thresholds'][thresh][model_type][key] += stats['thresholds'][thresh][model_type][key]

        # --- 关键：手动释放当前时刻的显存 ---
        del full_preds_t, full_unets_t, full_truths_t
        torch.cuda.empty_cache()

    # 5. 计算并打印所有时刻的平均指标 (基于累加器)
    print("\n====== 所有时刻平均指标 (Global Average) ======")
    total_count = global_accum['total_count']
    print(f"  MSE : {global_accum['total_mse'] / total_count:.4f}")
    print(f"  MAE : {global_accum['total_mae'] / total_count:.4f}")
    eps = 1e-6
    # 新增：初始化Fuxi表格数据存储
    fuxi_table_data = []  # 表格行数据
    fuxi_metrics = ["CSI", "HSS", "FAR", "BIAS", "PO"]  # Fuxi要展示的指标
    fuxi_threshold_vals = []  # 存储阈值，用于表头
    for t in cfg.thresholds:
        print(f"\n>>> 降水阈值 >= {t} mm/h")

        # U-Net Global
        u_stats = global_accum['thresholds'][t]['unet']
        u_tp, u_fn, u_fp, u_tn = u_stats['tp'], u_stats['fn'], u_stats['fp'], u_stats['tn']
        u_acc = (u_tp + u_tn) / (u_tp + u_fn + u_fp + u_tn + eps)
        u_precision = u_tp / (u_tp + u_fp + eps)
        u_recall = u_tp / (u_tp + u_fn + eps)
        u_f1 = 2 * (u_precision * u_recall) / (u_precision + u_recall + eps)
        u_po = u_fn / (u_tp + u_fn + eps)
        u_far = u_fp / (u_tp + u_fp + eps)
        u_csi = u_tp / (u_tp + u_fn + u_fp + eps)

        print(f"  [U-Net 分类表现]")
        print(f"    Accuracy : {u_acc:.4f}")
        print(f"    Precision: {u_precision:.4f}")
        print(f"    Recall   : {u_recall:.4f}")
        print(f"    F1 Score : {u_f1:.4f}")
        print(f"    PO (Miss): {u_po:.4f}")
        print(f"    FAR      : {u_far:.4f}")
        print(f"    CSI      : {u_csi:.4f}")

        # Fuxi Global
        r_stats = global_accum['thresholds'][t]['fuxi']
        r_tp, r_fn, r_fp, r_tn = r_stats['tp'], r_stats['fn'], r_stats['fp'], r_stats['tn']
        r_csi = r_tp / (r_tp + r_fn + r_fp + eps)
        r_hss = (2 * (r_tp * r_tn - r_fn * r_fp)) / (
                    (r_tp + r_fn) * (r_fn + r_tn) + (r_tp + r_fp) * (r_fp + r_tn) + eps)
        r_far = r_fp / (r_tp + r_fp + eps)
        r_bias = (r_tp + r_fp) / (r_tp + r_fn + eps)
        r_po = r_fn / (r_tp + r_fn + eps)

        print(f"  [Fuxi 回归表现]")
        print(f"    CSI      : {r_csi:.4f}")
        print(f"    HSS      : {r_hss:.4f}")
        print(f"    FAR      : {r_far:.4f}")
        print(f"    BIAS     : {r_bias:.4f}")
        print(f"    PO       : {r_po:.4f}")


        # 新增：收集当前阈值的Fuxi指标数据
        fuxi_threshold_vals.append(f">={t} mm/h")
        fuxi_table_data.append([
            r_csi, r_hss, r_far, r_bias, r_po
        ])

    # ---------------------- 新增：循环结束后打印Fuxi汇总表格 ----------------------
    from tabulate import tabulate  # 需先安装：pip install tabulate
    print("\n" + "=" * 60)
    print("Fuxi 各阈值指标汇总表")
    print("=" * 60)
    # 构建表格：行是指标名，列是所有阈值，值是对应指标结果
    table_rows = []
    for i, metric in enumerate(fuxi_metrics):
        # 遍历每个指标，收集所有阈值下的该指标值
        row = [metric] + [f"{fuxi_table_data[j][i]:.4f}" for j in range(len(fuxi_threshold_vals))]
        table_rows.append(row)

    # 打印表格（grid格式带边框，floatfmt保留4位小数）
    print(tabulate(
        table_rows,
        headers=["指标"] + fuxi_threshold_vals,  # 表头：指标 + 所有阈值
        tablefmt="grid",
        floatfmt=".4f"
    ))
    # -------------------------------------------------------------------------

    # ---------------------- 新增：打印短时强降水业务评分汇总 ----------------------
    def print_strong_stats(accum, title):
        print(f"====== {title}（短时强降水业务评分） ======")
        print(f"  强降水判识阈值 T : {strong_ts_threshold} mm/h")
        print(f"  扫描半径         : 20 km (~{strong_radius_px} px @ 4km)")
        print(f"  量级评分观测阈值 : >= {strong_me_obs_threshold} mm/h")

        TS, BIAS = strong_ts_bias_from_counts(accum['NA'], accum['NB'], accum['NC'])
        print(f"  TS              : {TS}")
        print(f"  BIAS            : {BIAS}")

        if accum['ME_COUNT'] == 0:
            print(f"  ME_SCORE        : nan (当前样本中无观测 >= {strong_me_obs_threshold} mm/h 的格点)")
        else:
            print(f"  ME_SCORE        : {accum['ME_SUM'] / accum['ME_COUNT']:.4f}")

    for t in range(cfg.out_len):
        print_strong_stats(strong_step_accums[t], f'第 {t + 1} 个预测时刻')

    print_strong_stats(strong_global_accum, '所有时刻平均')

































