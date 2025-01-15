import torch
import torch.nn as nn

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.loss_func = torch.nn.CrossEntropyLoss()


    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # contiguous：在不改变原数据的情况下，拷贝一份一样的数据
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # (8,2,68) -> (16,68),对应(batch_size,embedding_size)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits,div矩阵除法，除以0.01，扩大100倍
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),       # 每一行的平方和
            self.temperature)
        # for numerical stability
        # 返回输入张量给定维度上每行(dim=1)的最大值'logits_max'，并同时返回每个最大值的位置索引 '_'
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # 输入特征与最大值做差
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # 行扩大两倍，列也扩大两倍
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases,
        # scatter用于生成onehot向量,对角线全0，其他全1的128*128张量
        logits_mask = torch.scatter(
            # 生成与mask形状相同、元素全为1的张量
            torch.ones_like(mask),
            1,
            # 生成0到127的一维张量
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # mask == logita_mask 对角线全0，其他全1的128*128张量
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        # # ------------------------------------------------------------------
        # input_data = torch.div(contrast_feature, self.temperature)
        # target = labels.repeat(2, 1)
        # target = target.view(-1, 1).squeeze(1)
        # loss1 = self.Cross_loss(input_data, target)     # input_data输入的维度应该为(batch_size,num_class)
        # loss = loss1 + loss2
        # # ------------------------------------------------------------------
        return loss
