import torch
import torch.nn as nn
import torch.nn.functional as F


class  MODEL(nn.Module):
    def __init__(self,Igme,MoeFusion,classifier, Regressor,args):
    # def __init__(self,MoeFusion,classifier, Regressor, args):
        super(MODEL, self).__init__()
        self.Igme = Igme
        self.moeFusion = MoeFusion
        self.classifier = classifier
        self.Regressor = Regressor
        self.args = args
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))  #这表示 fusion_scale 不只是一个数字，它是一个可以通过反向传播（Backpropagation）更新的“权重”。


    def forward(self, x):  #[B, C, T, H, W]=(b,12,in_len,64,64)
        # # --- 准备数据进入 IGME 模块 ---
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 1, 2, 3, 4).reshape(B, C * T, H, W)
        x_enhanced_flat,pfim_loss = self.Igme(x_flat)
        H_new, W_new = x_enhanced_flat.shape[-2], x_enhanced_flat.shape[-1]
        x_enhanced = x_enhanced_flat.view(B, C, T, H_new, W_new).permute(0, 1, 2, 3, 4)

        # ######融合
        x_refined = self.moeFusion(x_enhanced)
        scale = torch.sigmoid(self.fusion_scale)  # 限制在 0~1
        x_fused = x_enhanced + (x_refined - x_enhanced) * scale

        ###########分类
        cls_logits = self.classifier(x_fused)
        cls_prob = torch.softmax(cls_logits,dim=1)


        ##########回归
        x_reg = torch.cat([x_fused, cls_prob], dim=1)

        pred_classes = torch.argmax(cls_prob, dim=1, keepdim=True)  # 把模型给出的**“每个类别的概率”转换成“最终预测的类别  (b,c,t,h,w)：(b,9,out_len,64,64)——>(b,1,out_len,64,64)
        mask_output = torch.where(pred_classes > 0, 1.0, 0.0)  # 属于第0类的是无降水，其他都是有降水的
        pred = self.Regressor(x_reg)
        pred = pred * mask_output


        if H!=H_new or W!=W_new:
            B, C, T, H_new, W_new = pred.shape  #[B, C, T, H, W]=(b,12,in_len,H_new,W_new)
            pred_reg_2d = pred.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H_new, W_new)  # 需要合并 B, T 维度才能用 interpolate
            pred = F.interpolate(pred_reg_2d, size=(H,W), mode='nearest')
            pred = pred.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)

            cls_2d = cls_logits.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H_new, W_new)
            cls_logits = F.interpolate(cls_2d, size=(H, W), mode='nearest')
            cls_logits = cls_logits.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)


        return pred, cls_logits,pfim_loss
        # return pred, cls_logits




