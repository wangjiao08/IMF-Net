import torch.nn.functional as F
import torch

# ==========================
# 回归模型评估指标
# ==========================
def pearson_correlation(y_true, y_pred):
    """皮尔逊相关系数"""
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    return cov / (torch.sqrt(var_true * var_pred) + 1e-7)

def mse(y_true, y_pred):
    """均方误差"""
    return F.mse_loss(y_pred, y_true)

def rmse(y_true, y_pred):
    """均方根误差"""
    return torch.sqrt(F.mse_loss(y_pred, y_true))

def mae(y_true, y_pred):
    """平均绝对误差"""
    return F.l1_loss(y_pred, y_true)


def pod(y_true, y_pred,t=1):
    #先二值化
    y_true = (y_true >= t).int()
    y_pred = (y_pred >= t).int()
    # 计算真阳性(TP)和实际正例总数(TP+FN)
    true_positives = torch.sum(y_true * y_pred)
    possible_positives = torch.sum(y_true)

    # 避免除零错误
    eps = torch.finfo(torch.float32).eps
    return true_positives / (possible_positives + eps)

#同TS
def csi(y_true, y_pred,t=1):
    y_true = (y_true >= t).int()
    y_pred = (y_pred >= t).int()

    # 计算混淆矩阵元素
    true_positives = torch.sum(y_true * y_pred)
    false_positives = torch.sum((1 - y_true) * y_pred)
    false_negatives = torch.sum(y_true * (1 - y_pred))

    # CSI = TP / (TP + FP + FN)
    return true_positives / (true_positives + false_positives + false_negatives + torch.finfo(torch.float32).eps)


def hss(y_true, y_pred, t=1):
    #预报比随机猜测好多少
    """
    Heidke Skill Score (HSS) - 海德克技巧评分
    范围: [-∞, 1]，1表示完美预测，0表示随机预测，负数表示比随机预测更差

    HSS = (TP + TN - E) / (Total - E)
    其中 E = [(TP+FN)*(TP+FP) + (FN+TN)*(FP+TN)] / Total
    """
    # 二值化
    y_true = (y_true >= t).int()
    y_pred = (y_pred >= t).int()

    # 计算混淆矩阵元素
    true_positives = torch.sum(y_true * y_pred)  # TP
    false_positives = torch.sum((1 - y_true) * y_pred)  # FP
    false_negatives = torch.sum(y_true * (1 - y_pred))  # FN
    true_negatives = torch.sum((1 - y_true) * (1 - y_pred))  # TN

    total = true_positives + false_positives + false_negatives + true_negatives

    # 避免除零错误
    eps = torch.finfo(torch.float32).eps

    # 计算随机预测的期望正确数
    expected_correct = ((true_positives + false_negatives) * (true_positives + false_positives) +
                        (false_negatives + true_negatives) * (false_positives + true_negatives)) / (total + eps)

    # 计算实际正确数
    actual_correct = true_positives + true_negatives

    # HSS公式
    numerator = actual_correct - expected_correct
    denominator = total - expected_correct

    return numerator / (denominator + eps)


def ssim(y_true, y_pred, data_range=1.0):
    """
    通用SSIM计算，支持任意形状，返回全局SSIM值
    """
    # 将输入展平为1D向量
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # 计算全局统计量
    mu_x = torch.mean(y_true_flat)
    mu_y = torch.mean(y_pred_flat)

    sigma_x = torch.var(y_true_flat, unbiased=False)
    sigma_y = torch.var(y_pred_flat, unbiased=False)
    sigma_xy = torch.mean((y_true_flat - mu_x) * (y_pred_flat - mu_y))

    # SSIM公式
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return numerator / (denominator + 1e-8)

#mar
def po(y_true, y_pred, t=1):

    y_true = (y_true >= t).int()
    y_pred = (y_pred >= t).int()
    false_negatives = torch.sum(y_true * (1 - y_pred))  # 实际发生但未预报
    possible_positives = torch.sum(y_true)  # 实际发生总数（TP + FN）
    eps = torch.finfo(torch.float32).eps
    return false_negatives / (possible_positives + eps)


def far(y_true, y_pred, t=1):

    """空报率（False Alarm Rate）= 空报数 / 预报发生数
    """
    y_true = (y_true >= t).int()
    y_pred = (y_pred >= t).int()
    false_positives = torch.sum((1 - y_true) * y_pred)  # 未发生但预报发生
    forecast_positives = torch.sum(y_pred)  # 预报发生总数（TP + FP）
    eps = torch.finfo(torch.float32).eps
    return false_positives / (forecast_positives + eps)


def bias(y_true, y_pred, t=1):
    """预报偏差（Bias）= 预报发生数 / 实际发生数"""
    y_true = (y_true >= t).int()
    y_pred = (y_pred >= t).int()
    forecast_positives = torch.sum(y_pred)  # 预报发生总数（TP + FP）
    possible_positives = torch.sum(y_true)  # 实际发生总数（TP + FN）
    eps = torch.finfo(torch.float32).eps
    return forecast_positives / (possible_positives + eps)


#unet
def accuracy(y_true, y_pred, t=1):
    """准确率（Accuracy）= 正确预测数 / 总样本数"""
    y_true = (y_true >= t).int()
    y_pred = (y_pred >= t).int()
    correct = torch.sum(y_true == y_pred)  # TP + TN
    total = torch.numel(y_true)  # 总样本数
    eps = torch.finfo(torch.float32).eps
    return correct / (total + eps)


def precision(y_true, y_pred, t=1):
    """精确率（Precision）= 真阳性 / 预报阳性总数（避免空报）"""
    y_true = (y_true >= t).int()
    y_pred = (y_pred >= t).int()
    true_positives = torch.sum(y_true * y_pred)  # TP
    predicted_positives = torch.sum(y_pred)  # TP + FP
    eps = torch.finfo(torch.float32).eps
    return true_positives / (predicted_positives + eps)


def recall(y_true, y_pred, t=1):
    """召回率（Recall）= 真阳性 / 实际阳性总数（避免漏报），与POD完全一致"""
    # 注：该指标与已有pod函数逻辑完全相同，仅名称不同（气象领域常用POD，机器学习常用Recall）
    return pod(y_true, y_pred, t=t)


def f1_score(y_true, y_pred, t=1):
    """F1分数（F1-Score）= 2*(精确率*召回率)/(精确率+召回率)，综合评估精确率和召回率"""
    prec = precision(y_true, y_pred, t=t)
    rec = recall(y_true, y_pred, t=t)
    eps = torch.finfo(torch.float32).eps
    return 2 * (prec * rec) / (prec + rec + eps)