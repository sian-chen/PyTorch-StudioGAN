# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/losses.py


import numpy as np

from utils.model_ops import snlinear, linear

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import autograd



# DCGAN loss
def loss_dcgan_dis(dis_out_real, dis_out_fake):
    device = dis_out_real.get_device()
    ones = torch.ones_like(dis_out_real, device=device, requires_grad=False)
    dis_loss = -torch.mean(nn.LogSigmoid()(dis_out_real) + nn.LogSigmoid()(ones - dis_out_fake))
    return dis_loss


def loss_dcgan_gen(gen_out_fake):
    return -torch.mean(nn.LogSigmoid()(gen_out_fake))


def loss_lsgan_dis(dis_out_real, dis_out_fake):
    dis_loss = 0.5*(dis_out_real - torch.ones_like(dis_out_real))**2 + 0.5*(dis_out_fake)**2
    return dis_loss.mean()


def loss_lsgan_gen(dis_out_fake):
    gen_loss = 0.5*(dis_out_fake - torch.ones_like(dis_out_fake))**2
    return gen_loss.mean()


def loss_hinge_dis(dis_out_real, dis_out_fake):
    return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))


def loss_hinge_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)


def loss_wgan_dis(dis_out_real, dis_out_fake):
    return torch.mean(dis_out_fake - dis_out_real)


def loss_wgan_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)


def latent_optimise(zs, fake_labels, gen_model, dis_model, conditional_strategy, latent_op_step, latent_op_rate,
                    latent_op_alpha, latent_op_beta, trans_cost, default_device):
    batch_size = zs.shape[0]
    for step in range(latent_op_step):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_() > 1 - latent_op_rate).to(default_device)
        z_gradients, z_gradients_norm = calc_derv(zs, fake_labels, dis_model, conditional_strategy, default_device, gen_model)
        delta_z = latent_op_alpha*z_gradients/(latent_op_beta + z_gradients_norm)
        zs = torch.clamp(zs + drop_mask*delta_z, -1.0, 1.0)

        if trans_cost:
            if step == 0:
                transport_cost = (delta_z.norm(2, dim=1)**2).mean()
            else:
                transport_cost += (delta_z.norm(2, dim=1)**2).mean()
            return zs, trans_cost
        else:
            return zs


def set_temperature(conditional_strategy, tempering_type, start_temperature, end_temperature, step_count, tempering_step, total_step):
    if conditional_strategy in ['ContraGAN', 'ContraGAN_plus']:
        if tempering_type == 'continuous':
            t = start_temperature + step_count*(end_temperature - start_temperature)/total_step
        elif tempering_type == 'discrete':
            tempering_interval = total_step//(tempering_step + 1)
            t = start_temperature + \
                (step_count//tempering_interval)*(end_temperature-start_temperature)/tempering_step
        else:
            t = start_temperature
    else:
        t = 'no'
    return t


class Cross_Entropy_loss(torch.nn.Module):
    def __init__(self, in_features, out_features, spectral_norm=True):
        super(Cross_Entropy_loss, self).__init__()

        if spectral_norm:
            self.layer =  snlinear(in_features=in_features, out_features=out_features, bias=True)
        else:
            self.layer =  linear(in_features=in_features, out_features=out_features, bias=True)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, embeds, labels):
        logits = self.layer(embeds)
        return self.ce_loss(logits, labels)


class Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, device, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix


    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)


    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
        instance_zone = torch.exp((self.remove_diag(similarity_matrix) - margin)/temperature)

        inst2proxy_positive = torch.exp((self.cosine_similarity(inst_embed, proxy) - margin)/temperature)
        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)
            inst2inst_positives = instance_zone*mask_4_remove_negatives

            numerator = inst2proxy_positive + inst2inst_positives.sum(dim=1)
        else:
            numerator = inst2proxy_positive

        denomerator = torch.cat([torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        criterion = -torch.log(temperature*(numerator/denomerator)).mean()
        return criterion


class Conditional_Contrastive_loss_plus(torch.nn.Module):
    def __init__(self, device, batch_size):
        super(Conditional_Contrastive_loss_plus, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix


    def remove_diag(self, M):
        h, w = M.shape
        assert h <= w, "w should be larger than h"
        mask = np.ones((h, w))
        mask[:,:h] -= np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)


    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def forward(self, inst_embed, proxy, negative_mask, labels, temperature):
        sim_mat = self.calculate_similarity_matrix(inst_embed, inst_embed)/temperature
        i2p_pos = self.cosine_similarity(inst_embed, proxy)/temperature
        ip_mat = torch.cat([sim_mat, i2p_pos.unsqueeze(dim=1)], dim=1)
        ip_max, _ = torch.max(ip_mat, dim=1, keepdim=True)
        ip_mat = ip_mat - ip_max.detach()

        mask_4_rmv_neg = torch.cat([self.remove_diag(negative_mask[labels]),
                                    torch.ones(self.batch_size, 1).to(self.device)], dim=1)
        ip_mat_rmv_self = self.remove_diag(ip_mat)

        log_prob = ip_mat_rmv_self - torch.log(torch.exp(ip_mat_rmv_self).sum(1, keepdim=True))
        log_prob_pos = ((log_prob*mask_4_rmv_neg).sum(1))/(mask_4_rmv_neg.sum(1))
        criterion = -log_prob_pos.mean()
        return criterion

"""
class Conditional_Contrastive_loss_plus(torch.nn.Module):
    def __init__(self, device, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss_plus, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix


    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)


    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        p2i_similarity_matrix = self.calculate_similarity_matrix(proxy, inst_embed)
        i2i_similarity_matrix = self.calculate_similarity_matrix(inst_embed, inst_embed)
        p2i_similarity_zone = torch.exp((p2i_similarity_matrix - margin)/temperature)
        i2i_similarity_zone = torch.exp((i2i_similarity_matrix - margin)/temperature)

        mask_4_remove_negatives = negative_mask[labels]
        p2i_positives = p2i_similarity_zone*mask_4_remove_negatives
        i2i_positives = i2i_similarity_zone*mask_4_remove_negatives

        p2i_numerator = p2i_positives.sum(dim=1)
        i2i_numerator = i2i_positives.sum(dim=1)
        p2i_denomerator = p2i_similarity_zone.sum(dim=1)
        i2i_denomerator = i2i_similarity_zone.sum(dim=1)

        p2i_contra_loss = -torch.log(temperature*(p2i_numerator/p2i_denomerator)).mean()
        i2i_contra_loss = -torch.log(temperature*(i2i_numerator/i2i_denomerator)).mean()
        return p2i_contra_loss + i2i_contra_loss
"""

class Proxy_NCA_loss(torch.nn.Module):
    def __init__(self, device, embedding_layer, num_classes, batch_size):
        super(Proxy_NCA_loss, self).__init__()
        self.device = device
        self.embedding_layer = embedding_layer
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


    def _get_positive_proxy_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        rvs_one_hot_target = np.ones([self.num_classes, self.num_classes]) - np.eye(self.num_classes)
        rvs_one_hot_target = rvs_one_hot_target[labels]
        mask = torch.from_numpy((rvs_one_hot_target)).type(torch.bool)
        return mask.to(self.device)


    def forward(self, inst_embed, proxy, labels):
        all_labels = torch.tensor([c for c in range(self.num_classes)]).type(torch.long).to(self.device)
        positive_proxy_mask = self._get_positive_proxy_mask(labels)
        negative_proxies = torch.exp(torch.mm(inst_embed, self.embedding_layer(all_labels).T))*positive_proxy_mask

        inst2proxy_positive = torch.exp(self.cosine_similarity(inst_embed, proxy))
        numerator = inst2proxy_positive
        denomerator = negative_proxies.sum(dim=1)
        criterion = -torch.log(numerator/denomerator).mean()
        return criterion


class NT_Xent_loss(torch.nn.Module):
    def __init__(self, device, batch_size, use_cosine_similarity=True):
        super(NT_Xent_loss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")


    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity


    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v


    def forward(self, zis, zjs, temperature):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)


def calc_derv4gp(netD, conditional_strategy, real_data, fake_data, real_labels, device):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(batch_size,c,h,w)
    alpha = alpha.to(device)

    real_data = real_data.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
            disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy == 'ACGAN':
        _, disc_interpolates = netD(interpolates, real_labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv4dra(netD, conditional_strategy, real_data, real_labels, device):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.to(device)

    real_data = real_data.to(device)
    differences  = 0.5*real_data.std()*torch.rand(real_data.size()).to(device)

    interpolates = real_data + (alpha*differences)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
            disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy == 'ACGAN':
        _, disc_interpolates = netD(interpolates, real_labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv(inputs, labels, netD, conditional_strategy, device, netG=None):
    zs = autograd.Variable(inputs, requires_grad=True)
    fake_images = netG(zs, labels)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
        dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy == 'ACGAN':
        _, dis_out_fake = netD(fake_images, labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=dis_out_fake, inputs=zs,
                              grad_outputs=torch.ones(dis_out_fake.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients_norm = torch.unsqueeze((gradients.norm(2, dim=1) ** 2), dim=1)
    return gradients, gradients_norm


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

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
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        import pdb; pdb.set_trace()
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        import pdb; pdb.set_trace()
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
