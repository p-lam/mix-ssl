from numpy import kaiser
import torch.nn as nn 
from torch.nn import functional as F
import torch 
import modules.resnet as RN 
from functools import partial
from torchvision.models import resnet
import modules.resnet as RN 

# class ModelBase(nn.Module):
#     """
#     Common CIFAR ResNet recipe.
#     Comparing with ImageNet ResNet recipe, it:
#     (i) replaces conv1 with kernel=3, str=1
#     (ii) removes pool1
#     """
#     def __init__(self, dataset, depth, num_classes, feature_dim=128, bn_splits=16):
#         super(ModelBase, self).__init__()

#         # use split batchnorm
#         norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
#         self.net = RN.ResNet(dataset, depth, num_classes=feature_dim, norm_layer=norm_layer)

#     def forward(self, x):
#         x, x2 = self.net(x)
#         # note: not normalized here
#         return x, x2

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        # x = self.net(x)
        pre_pool_feature = self.net[:7](x)
        final_feature = self.net[7:](pre_pool_feature)
        # note: not normalized here
        return final_feature, pre_pool_feature

class DetMix(nn.Module):
    """
    DetMix
    modified from MoCo
    """
    def __init__(self, dataset, depth, num_classes, dim=128, K=4096, m=0.99, T=0.1, bn_splits=8, symmetric=True):
        super(DetMix, self).__init__()
        self.depth = depth 
        self.num_classes = num_classes
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        # self.encoder_q = ModelBase(dataset, depth, num_classes, feature_dim=dim, bn_splits=bn_splits)
        # self.encoder_k = ModelBase(dataset, depth, num_classes, feature_dim=dim, bn_splits=bn_splits)
        self.encoder_q = ModelBase(feature_dim=dim, arch='resnet18', bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch='resnet18', bn_splits=bn_splits)

        # project resnet feat
        self.projection = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("local_queue", torch.randn(dim, K))
        self.local_queue = nn.functional.normalize(self.local_queue, dim=0)
        self.register_buffer("local_ptr", torch.zeros(1, dtype=torch.long))

        # self.register_buffer("global_queue", torch.randn(dim, K))
        # self.global_queue = nn.functional.normalize(self.global_queue, dim=0)
        # self.register_buffer("global_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]
    
    def crop_and_upsample(self, final_feature, boxes):
        final_feature = self.projection(final_feature)
        final_feature = nn.Upsample(size=(32,32), mode="bilinear")(final_feature)
        bbx1, bbx2, bby1, bby2 = boxes

        # compute avg feature across cropped region
        crop_feature = final_feature[:,:, bbx1:bbx2, bby1:bby2]
        avg_crop_feature = nn.AdaptiveAvgPool2d((1,1))(crop_feature).view(-1, 128)
        avg_crop_feature = nn.functional.normalize(avg_crop_feature, dim=1)

        mask = torch.ones_like(final_feature)
        mask[:,:, bbx1:bbx2, bby1:bby2] = 0

        uncrop_feature = final_feature * mask
        # uncrop_feature = nn.AdaptiveAvgPool2d((1,1))(global_feature).view(-1, 128)
        # uncrop_feature = nn.functional.normalize(global_feature, dim=1)
        area_total = uncrop_feature.size()[-1] * uncrop_feature.size()[-2]
        area_crop = (bbx2 - bbx1) * (bby2 - bby1)
        area_uncrop = area_total - area_crop
        uncrop_feature = uncrop_feature.view(uncrop_feature.size(0), uncrop_feature.size(1), -1) # N x D x AREA_TOTAL
        uncrop_feature_sum = uncrop_feature.sum(dim=2)
        avg_uncrop_feature = uncrop_feature_sum / area_uncrop
        avg_uncrop_feature = nn.functional.normalize(avg_uncrop_feature, dim=1)

        return avg_crop_feature, avg_uncrop_feature

    def contrastive_loss(self, im_q, im_k, im1_mixed, im2_mixed, im1_mixed_re, im2_mixed_re, boxes, boxes2):

        # compute query features
        q, _ = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        q_mixed, q_feature = self.encoder_q(im1_mixed)
        q_mixed = nn.functional.normalize(q_mixed, dim=1)
        q_mixed_flip, mixed_feature = self.encoder_q(im1_mixed_re)
        q_mixed_flip = nn.functional.normalize(q_mixed_flip, dim=1)
        
        crop_feature, global_feature = self.crop_and_upsample(q_feature, boxes)
        crop_feature_flip, global_feature_flip = self.crop_and_upsample(mixed_feature, boxes)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k, _ = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            k_mixed, k_features = self.encoder_k(im2_mixed) 
            k_crop, k_global = self.crop_and_upsample(k_features, boxes2)

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_pos_m = torch.einsum('nc,nc->n', [q_mixed, k]).unsqueeze(-1)
        l_pos_ll = torch.einsum('nc,nc->n', [crop_feature, k_crop]).unsqueeze(-1)
        l_pos_gg = torch.einsum('nc,nc->n', [global_feature, k]).unsqueeze(-1)
        # l_pos_gl = torch.einsum('nc,nc->n', [crop_feature, k_global]).unsqueeze(-1)

        l_pos_m_re = torch.einsum('nc,nc->n', [q_mixed_flip, k]).unsqueeze(-1)
        l_pos_ll_re = torch.einsum('nc,nc->n', [crop_feature_flip, k_crop]).unsqueeze(-1)
        l_pos_gg_re = torch.einsum('nc,nc->n', [global_feature_flip, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_neg_m = torch.einsum('nc,ck->nk', [q_mixed, self.queue.clone().detach()])
        l_neg_ll = torch.einsum('nc,ck->nk', [crop_feature, self.local_queue.clone().detach()])
        l_neg_gg = torch.einsum('nc,ck->nk', [global_feature, self.queue.clone().detach()])
        # l_neg_gl = torch.einsum('nc,ck->nk', [crop_feature, self.global_queue.clone().detach()])

        l_neg_m_re = torch.einsum('nc,ck->nk', [q_mixed_flip, self.queue.clone().detach()])
        l_neg_ll_re = torch.einsum('nc,ck->nk', [crop_feature_flip, self.local_queue.clone().detach()])
        l_neg_gg_re = torch.einsum('nc,ck->nk', [global_feature_flip, self.queue.clone().detach()])
       
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_gg = torch.cat([l_pos_gg, l_neg_gg], dim=1)
        logits_ll = torch.cat([l_pos_ll, l_neg_ll], dim=1)
        # logits_gl = torch.cat([l_pos_gl, l_neg_gl], dim=1)
        logits_m = torch.cat([l_pos_m, l_neg_m], dim=1)

        logits_gg_flip = torch.cat([l_pos_gg_re, l_neg_gg_re], dim=1)
        logits_ll_flip = torch.cat([l_pos_ll_re, l_neg_ll_re], dim=1)
        logits_m_flip = torch.cat([l_pos_m_re, l_neg_m_re], dim=1)

        # apply temperature
        logits /= self.T
        logits_gg /= self.T
        logits_ll /= self.T
        # logits_gl /= self.T
        logits_m /= self.T

        logits_gg_flip /= self.T
        logits_ll_flip /= self.T
        logits_m_flip /= self.T 

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)
        loss_m1 = nn.CrossEntropyLoss().cuda()(logits_m, labels)
        loss_m2 = nn.CrossEntropyLoss().cuda()(logits_m_flip, labels)
        loss_gg1 = nn.CrossEntropyLoss().cuda()(logits_gg, labels)
        loss_gg2 = nn.CrossEntropyLoss().cuda()(logits_gg_flip, labels)
        loss_ll1 = nn.CrossEntropyLoss().cuda()(logits_ll, labels)
        loss_ll2 = nn.CrossEntropyLoss().cuda()(logits_ll_flip, labels)
        # loss_gl1 = nn.CrossEntropyLoss().cuda()(logits_gl, labels)


        return loss, q, k, loss_m1, loss_m2, loss_gg1, loss_gg2, loss_ll1, loss_ll2, k_crop
        # return loss, q, k, loss_m1


    def forward(self, im1, im2, im1_mixed, im2_mixed, im1_mixed_re, im2_mixed_re, lam, boxes, boxes2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        
        # compute loss
        if self.symmetric:  # symmetric loss
            # loss_12, q1, k2, loss_m11, loss_m12 = self.contrastive_loss(im1, im2, im1_mixed, im1_mixed_re)
            # loss_21, q2, k1, loss_m21, loss_m22 = self.contrastive_loss(im2, im1, im1_mixed, im1_mixed_re)
            # loss = loss_12 + loss_21 + lam*loss_m11 + (1-lam)*loss_m12 + lam*loss_m21 + (1-lam)*loss_m22
            # k = torch.cat([k1, k2], dim=0)
            pass
        else:  # asymmetric loss
            loss_0, q, k, loss_m1, loss_m2, loss_gg1, loss_gg2, loss_ll1, loss_ll2, k_crop = self.contrastive_loss(im1, im2, im1_mixed, im2_mixed, im1_mixed_re, im2_mixed_re, boxes, boxes2)
            # loss_0, q, k, loss_m1 = self.contrastive_loss(im1, im2, im1_mixed, im2_mixed, boxes, boxes2)

            # loss = loss_0 + lam*loss_m1 + (1-lam)*loss_m2
            loss = loss_0 + lam*(loss_m1 + loss_gg1 + loss_ll1) + (1-lam)*(loss_m2 + loss_gg2 + loss_ll2)
            # loss = loss_0 + (1/3)*lam*(loss_m1 + loss_gg1 + loss_l11) + (1/3)*(1-lam)*(loss_m2 + loss_gg2 + loss_ll2)

        self._dequeue_and_enqueue(k, self.queue, self.queue_ptr)
        self._dequeue_and_enqueue(k_crop, self.local_queue, self.local_ptr)
        # self._dequeue_and_enqueue(k_global, self.global_queue, self.global_ptr)

        return loss

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)
