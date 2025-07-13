import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    """
    Combines CrossEntropy with KL Divergence loss between soft logits.
    T: Temperature
    alpha: Weight for CE vs KL loss
    """
    ce_loss = F.cross_entropy(student_logits, labels)

    kl_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    total_loss = alpha * ce_loss + (1 - alpha) * kl_loss
    return total_loss
