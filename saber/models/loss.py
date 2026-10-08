import torch
import torch.nn as nn
import torch.nn.functional as F


class SABERLoss(nn.Module):
    def __init__(self):
        super(SABERLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.alphas = [1.0, 2.0, 4.0, 8.0, 16.0]

    def get_task_loss(self, proportions_pred, proportions_true):
        """
        Source-domain supervised loss.

        Args:
            proportions_pred: [B, C]
            proportions_true: [B, C]

        Returns:
            prop_loss: supervised proportion loss
        """
        return self.l1_loss(proportions_pred, proportions_true)

    def _normalize_probs(self, probs, eps=1e-8):
        probs = probs.clamp_min(eps)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)

    def get_directional_kl_loss(self, anchor_probs, student_probs):
        """
        Directional teacher-to-student KL consistency.

        The anchor branch is detached so consistency gradients only update the
        student path.
        """
        anchor = self._normalize_probs(anchor_probs.detach())
        student = self._normalize_probs(student_probs)
        return F.kl_div(student.log(), anchor, reduction="batchmean")

    def get_latent_consistency_loss(self, anchor_z, student_z):
        """
        Directional latent consistency with a detached anchor.
        """
        dim = max(student_z.size(1), 1)
        anchor = anchor_z.detach()
        return (student_z - anchor).pow(2).sum(dim=1).mean() / float(dim)

    def get_source_consistency_loss(
        self,
        p_clean,
        p_obs,
        z_clean,
        z_obs,
        lambda_latent,
    ):
        pred_cons = self.get_directional_kl_loss(p_clean, p_obs)
        latent_cons = self.get_latent_consistency_loss(z_clean, z_obs)
        return pred_cons + lambda_latent * latent_cons

    def _compute_kernel(self, x, y):
        dim = max(x.size(1), 1)
        x_norm = x.pow(2).sum(dim=1, keepdim=True)
        y_norm = y.pow(2).sum(dim=1, keepdim=True).transpose(0, 1)
        kernel_input = (x_norm + y_norm - 2.0 * x.matmul(y.transpose(0, 1))).clamp_min(0.0)
        kernel_input = kernel_input / float(dim)

        kernel_matrix = torch.zeros_like(kernel_input)
        for alpha in self.alphas:
            kernel_matrix = kernel_matrix + torch.exp(-kernel_input / alpha)
        return kernel_matrix

    def get_conditional_local_mmd_weights(self, proportions_src, proportions_tgt, tau=10.0):
        """
        Source-neighborhood weights for each target sample.

        Target proportions are detached: they define local neighborhoods but
        are not used as target pseudo-labels.
        """
        src = self._normalize_probs(proportions_src.detach())
        tgt = self._normalize_probs(proportions_tgt.detach())
        distances = torch.cdist(src, tgt, p=2).pow(2)
        return F.softmax(-float(tau) * distances, dim=0)

    def get_conditional_local_mmd_loss(
        self,
        z_src_prop,
        z_tgt_prop,
        proportions_src,
        proportions_tgt,
        tau=10.0,
    ):
        """
        Conditional local MMD between source and target proportion latents.

        Each target sample is aligned to a soft source neighborhood whose
        composition labels are close to the target sample's detached current
        proportion prediction.
        """
        alpha = self.get_conditional_local_mmd_weights(
            proportions_src,
            proportions_tgt,
            tau=tau,
        )
        k_ss = self._compute_kernel(z_src_prop, z_src_prop)
        k_st = self._compute_kernel(z_src_prop, z_tgt_prop)
        k_tt = self._compute_kernel(z_tgt_prop, z_tgt_prop).diag()

        source_term = (alpha * k_ss.matmul(alpha)).sum(dim=0)
        cross_term = (alpha * k_st).sum(dim=0)
        local_mmd = source_term + k_tt - 2.0 * cross_term
        return local_mmd.clamp_min(0.0).mean()
