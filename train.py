from torch.utils.data import DataLoader
import time
import matplotlib
matplotlib.use('Agg')
from config import parse_args
from models.IGME import IGME
from models.moefusion import MOEFUSION
from models.model import MODEL
from models.Transunet import Transunet
from models.Swintransformer import Swintransformer
from util.loss import *
from util.focal_loss_new import focal_loss
from optim import WarmupCosineScheduler
from eval import *
from util.load_data_repro import load_cache_data
import os
from datetime import datetime, timedelta
from sample_full import SAMPLE_FULL
from sample import SAMPLE
import random

def fix_random(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)  # 固定random.random()生成的随机数
    np.random.seed(seed)  # 固定np.random()生成的随机数
    torch.manual_seed(seed)  # 固定CPU生成的随机数
    torch.cuda.manual_seed(seed)  # 固定GPU生成的随机数-单卡
    torch.cuda.manual_seed_all(seed)  # 固定GPU生成的随机数-多卡
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      #False会避免 cuDNN 每次根据测速选择不同卷积算法
    torch.backends.cudnn.enabled = True        #T会让卷积等操作尽量走确定性算法
    torch.use_deterministic_algorithms(True, warn_only=True)



def compute_validation_loss(args, model, loader, train_criterion,
                            cls_criterion, lambda_cls,
                            lambda_pfim,
                            heavy_criterion,lambda_heavy):
    val_criterion = FACL(args.n_steps).to(args.device)
    val_criterion.step = train_criterion.step

    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch[:, :args.in_len, : , :, :].permute(0, 2, 1, 3, 4)
            gts = batch[:, -args.out_len:, 3:4, :, :].permute(0, 2, 1, 3, 4)

            imgs, gts = map(lambda x: x.to(args.device), [imgs, gts])

            preds,cls_logits,pfim_loss = model(imgs)
            # preds,cls_logits = model(imgs)

            loss_facl = val_criterion(preds, gts)
            loss_cls = cls_criterion(cls_logits, gts)
            loss_heavy = heavy_criterion(preds, gts)

            loss = loss_facl + lambda_cls * loss_cls + lambda_pfim * pfim_loss + lambda_heavy * loss_heavy
            # loss = loss_facl + lambda_cls * loss_cls + + lambda_heavy * loss_heavy

            total_loss += loss.item()
            total_batches += 1

    model.train()

    return total_loss / max(total_batches, 1)

def save_loss_curve(train_steps, train_losses, val_steps, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    if train_steps:
        plt.plot(train_steps, train_losses, label='Train Loss', linewidth=1.5)
    if val_steps:
        plt.plot(val_steps, val_losses, label='Val Loss', linewidth=1.5)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training / Validation Loss')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_Model(args):

    def seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    fix_random(args.seed)


    train_dataset, valid_dataset, test_dataset = load_cache_data(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              worker_init_fn=seed_worker,generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=40, shuffle=False, num_workers=4,
                              worker_init_fn=seed_worker,generator=g
                              )
    test_loader  = DataLoader(test_dataset,  batch_size=40, shuffle=False, num_workers=4,
                              worker_init_fn=seed_worker,generator=g
                              )


    #放大增强模块
    total_in_channels = args.in_chans * args.input_length
    args.scale = 2
    Igme = IGME(in_channels=total_in_channels, scale=args.scale).to(args.device)
    #
    # # 融合模块
    MoeFusion = MOEFUSION(in_channels=args.in_chans, in_timesteps=args.input_length).to(args.device)

    #分类
    classifier = Transunet(
        args,
        in_channels=args.in_chans,
        input_timesteps=args.input_length,
        out_channels=9,
        output_timesteps=args.output_length,
        img_size=args.hw * args.scale
        # img_size=args.hw
    ).to(args.device)

    #回归
    Regressor  = Swintransformer(
        # img_size=(args.input_length, args.hw, args.hw),
        img_size=(args.input_length, args.hw * args.scale, args.hw * args.scale),
        patch_size=(args.input_length, 4, 4), in_chans=args.in_chans+9, out_chans=1,
        embed_dim=768, num_groups=32, num_heads=8, window_size=8,out_timesteps=args.output_length).to(args.device)


    model = MODEL(Igme,MoeFusion,classifier, Regressor,args).to(args.device)
    # model = MODEL(MoeFusion, classifier, Regressor,args).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    scheduler = WarmupCosineScheduler(optimizer, args.n_steps, base_lr=args.lr).get_scheduler()

    criterion = FACL(args.n_steps).to(args.device)

    cls_criterion = focal_loss(args)
    lambda_cls = 0.5

    lambda_pfim = 1e-5

    heavy_criterion = HeavyRainWeightedMAELoss(args).to(args.device)
    lambda_heavy = 0.05

    step_cnt = 0
    train_loss = 0.
    start_time = time.time()

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    # # 开始训练
    # print('Start training...')
    # model.train()
    # while step_cnt < args.n_steps:
    #
    #     for  batch in tqdm(iter(train_loader)):
    #         imgs = batch[:,:args.in_len, :,:,:].permute(0, 2, 1, 3, 4)    #[B, T, C, H, W] ——> [B, C, T, H, W]=(b,12,in_len,64,64)  前两个时刻的12个通道
    #         gts = batch[:,-args.out_len:,3:4,:,:].permute(0, 2, 1, 3, 4)  #[B, T, C, H, W] ——> [B, C, T, H, W]=(b,1,in_len,64,64)  未来两个时刻的降水通道
    #
    #         optimizer.zero_grad()
    #
    #         imgs, gts = map(lambda x: x.to(args.device), [imgs, gts])
    #
    #         preds,cls_logits,pfim_loss = model(imgs)
    #         # preds,cls_logits = model(imgs)
    #
    #         loss_facl = criterion(preds , gts )
    #         loss_cls = cls_criterion(cls_logits, gts)
    #         loss_heavy = heavy_criterion(preds, gts)
    #
    #         loss = loss_facl + lambda_cls * loss_cls + lambda_pfim * pfim_loss + lambda_heavy * loss_heavy
    #         # loss = loss_facl  + lambda_cls * loss_cls + lambda_heavy * loss_heavy
    #
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         train_loss += loss.item()
    #
    #         step_cnt += 1
    #
    #         if step_cnt % args.log_freq == 0:
    #             avg_train_loss = train_loss / args.log_freq
    #
    #             # elapsed_time = time.time() - start_time
    #             # print(
    #             #     f'Step #{step_cnt}: total_loss={train_loss / args.log_freq:.8f} '
    #             #     f'facl={loss_facl.item():.8f} '
    #             #     f'cls={loss_cls.item():.8f} ',
    #             #     f'pfim_loss={pfim_loss.item():.8f}',
    #             #     f'heavy={loss_heavy.item():.8f} ',
    #             #     f'{elapsed_time:.2f} seconds\n'
    #             # )
    #             train_steps.append(step_cnt)
    #             train_losses.append(avg_train_loss)
    #             train_loss = 0.
    #
    #         # if step_cnt % args.save_freq == 0:
    #         #     torch.save(model.state_dict(), args.model_save_best)
    #         #
    #         # if step_cnt % args.val_freq == 0:
    #         #     val_loss = compute_validation_loss(
    #         #         args, model, valid_loader, criterion,
    #         #         cls_criterion, lambda_cls,
    #         #         lambda_pfim,
    #         #         heavy_criterion,lambda_heavy)
    #         #
    #         #     val_steps.append(step_cnt)
    #         #     val_losses.append(val_loss)
    #         #     print(f'Step #{step_cnt}: val_loss is {val_loss:.8f}\n')
    #         #     save_loss_curve(train_steps, train_losses, val_steps, val_losses, args.loss_curve_png)
    #         #
    #         # if step_cnt == args.n_steps:
    #         #     elapsed_time = time.time() - start_time
    #         #     print(f'Final Elapsed Time: {elapsed_time:.2f} seconds\n')
    #         #     break
    # save_loss_curve(train_steps, train_losses, val_steps, val_losses, args.loss_curve_png)
    # torch.save(model.state_dict(), args.model_save_best)


    # print('Start visualization...')
    # time_list = ['2021061009']
    # for label_time in time_list:
    #     SAMPLE(args, model, label_time=label_time)
    # torch.cuda.empty_cache()
    #
    # print('Start testing...')
    # evaluation(args, test_loader, model)


    # print('SHAP...')
    # from shap_sample import SHAP_SAMPLE
    # path = f"{args.home}/64_4km/normalization_64_npy/test_new/"
    # # 遍历所有npy文件
    # for f in sorted(os.listdir(path)):
    #     if f.endswith(".npy") and len(f) == 14:
    #         t = datetime.strptime(f[:10], "%Y%m%d%H")
    #         # 生成连续4个时间
    #         times = [t + timedelta(hours=i) for i in range(4)]
    #         files = [path + x.strftime("%Y%m%d%H.npy") for x in times]
    #         # 4个都存在 → 输出label_time
    #         if all(os.path.exists(p) for p in files):
    #             label_time = times[-1].strftime("%Y%m%d%H")
    #             SHAP_SAMPLE(
    #                 args,
    #                 model,
    #                 label_time=label_time,
    #                 rain_threshold=0.1,
    #                 baseline_split='train',
    #                 baseline_max_files=48,
    #                 case_batch_size=16,   #一次处理多少个 64×64 小块
    #                 subset_batch_size=4   #一次处理多少个 SHAP 通道组合
    #             )
    #             torch.cuda.empty_cache()

    from shap_sample import SHAP_SAMPLE
    SHAP_SAMPLE(
        args,
        model,
        label_time='2021061009',
        rain_threshold=0.1,
        baseline_split='train',
        baseline_max_files=48,
        case_batch_size=16,  # 一次处理多少个 64×64 小块
        subset_batch_size=4  # 一次处理多少个 SHAP 通道组合
    )

    # print("可视化全部测试集_____________________________________________________")
    # # 你的路径
    # path = f"{args.home}/64_4km/normalization_64_npy/test_new/"
    # model.load_state_dict(torch.load(args.model_save_best))
    # # 遍历所有npy文件
    # for f in sorted(os.listdir(path)):
    #     if f.endswith(".npy") and len(f) == 14:
    #         t = datetime.strptime(f[:10], "%Y%m%d%H")
    #         # 生成连续4个时间
    #         times = [t + timedelta(hours=i) for i in range(4)]
    #         files = [path + x.strftime("%Y%m%d%H.npy") for x in times]
    #         # 4个都存在 → 输出label_time
    #         if all(os.path.exists(p) for p in files):
    #             label_time = times[-1].strftime("%Y%m%d%H")
    #             SAMPLE_FULL(args, model, label_time)
    # print("DONE")




if __name__ == '__main__':
    # for i in range(1000,2000):
        i = 13
        args = parse_args()
        args.seed = i
        args.model_save_best = f'{args.home}/64_4km/Transunet_Swintransformer/result/A_3_14_{i}.pth'
        args.loss_curve_png = f'{args.home}/64_4km/Transunet_Swintransformer/result/A_3_14_{i}.png'
        args.gpu_id = '0'
        args.device_ids = [int(i) for i in args.gpu_id.split(',')]
        args.device = f'cuda:{args.device_ids[0]}'
        # print(f"正在训练第{i}个模型")
        train_Model(args)


