import torch
import torch.nn.functional as F

def btm_1_loss(f, labels):
    labels = torch.min(labels, dim=-1)[1]
    return F.nll_loss(F.log_softmax(f,1), labels)

def forward_loss(f, labels, T):
    # q = torch.mm(F.softmax(f, dim=1), T).clamp(1e-8,1-1e-8)
    # loss = (-q.log() * labels).sum(-1).mean()

    # Tinv = torch.inverse(T)
    Tinv = T
    # q = F.softmax(f, dim=1) - 1
    q = F.log_softmax(f, dim=1)
    loss = ((-q.matmul(Tinv.transpose(1,0))) * labels).sum(-1).mean()

    # loss += torch.special.entr(F.softmax(f,dim=1)).sum(-1).mean() * 0.1

    return loss

def _forward_loss(f, labels, T):
    # ceil = ((torch.special.entr(T)) / T.sum(0).view(1,-1)).sum(0).mean()
    ceil = ((torch.special.entr(T)) / T.sum(0).view(1,-1)).sum(0)

    q = torch.mm(F.softmax(f, dim=1), T).clamp(1e-8,1-1e-8)
    loss = (-q.log() * labels).sum(-1).mean()

    return loss

    # loss = (-q.log() * labels).sum(0)
    # ceil *= labels.sum(0)

    # return ((loss - ceil).abs() + ceil).sum() / labels.size(0)

    # loss = (-q.log() * labels).sum(-1).mean(0)
    # ceil = torch.special.entr(T).sum() / labels.size(1)

    # return (loss-ceil).abs() + ceil

def ure_ga_loss(f, labels, K, label_dist):
    logprob = F.log_softmax(f,1)
    l = (-logprob * labels)

    labels_count = labels.sum(0)
    l_sum = l.sum(0)

    loss = torch.zeros_like(l_sum)
    idx = labels_count > 0
    loss[idx] = -(K-1) * l_sum[idx] / labels_count[idx] * label_dist[idx]
    for j in range(K):
        if labels_count[j] > 0:
            loss += (-logprob * labels[:,j].unsqueeze(1)).sum(0) / labels_count[j] * label_dist[j]

    if loss.min() > 0:
        return loss.sum()
    else:
        return F.relu(-loss).sum()
    return loss

def scl_log_loss(f, labels): #scl-nl
    # l = F.log_softmax(-f,1)

    # pos_sm = F.softmax(f,1)
    # neg_sm = 1 / F.softmax(-f,1)
    # w = F.softmax(neg_sm+1,1) * pos_sm + 1e-6

    # l = l * w

    l = (1-F.softmax(f,1)).clamp(1e-8,1-1e-8).log()
    # l = -(1-(1-F.softmax(f,1)).pow(0.75))/0.75
    # l = -F.softmax(f,1)

    # return F.nll_loss(l, labels.view(-1))
    # return (-l * F.one_hot(labels,K).sum(1).float()).sum(-1).mean()
    return (-l * labels).sum(-1).mean()

def scl_lin_loss(f, labels):
    # l = (1-F.softmax(f,1)).clamp(1e-8,1-1e-8).log()
    l = -F.softmax(f,1)

    # return F.nll_loss(l, labels.view(-1))
    # return (-l * F.one_hot(labels,K).sum(1).float()).sum(-1).mean()

    return (-l * labels).sum(-1).mean()

    # labels = (torch.ones_like(labels) - labels) / labels.size(1)
    # return (-l - labels).abs().sum(-1).mean()

def non_k_softmax_loss(f, labels):
    Q_1 = 1 - F.softmax(f, 1)
    Q_1 = F.log_softmax(Q_1, 1)
    return (-Q_1 * labels).sum(-1).mean()

def w_loss(f, labels, K):
    loss_class = non_k_softmax_loss(f=f, labels=labels)
    loss_w = w_loss_p(f=f, K=K, labels=labels)
    final_loss = loss_class + loss_w
    return final_loss

def w_loss_p(f, K, labels):
    Q_1 = 1-F.softmax(f, 1)
    Q = F.log_softmax(Q_1, 1)
    w_1 = torch.mul(Q_1 / (K-1), Q)
    return (-w_1 * labels).sum(-1).mean()

# def pc_loss(f, labels, K):
#     labels = torch.multinomial(labels, 1)
#     fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
#     loss_matrix = torch.sigmoid(-1.*(f-fbar)) # multiply -1 for "complementary"
#     M1, M2 = K*(K-1)/2, K-1
#     pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
#     return pc_loss

# def pc_loss(f, labels, K):
#     f = f - f * labels
#     loss = torch.sigmoid(-1 * f).sum(dim=1).mean() - 0.5
# 
#     return loss

def pc_loss(f, labels, K):
    # labels = labels.max(dim=-1)[1]
    pc_loss = 0
    for k in range(K):
        # fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
        fbar = f[:,k].view(-1,1)
        loss_matrix = torch.sigmoid(-1.*(f-fbar)) # multiply -1 for "complementary"
        # loss_matrix = (1 + (fbar - f).exp()).log()
        # pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
        pc_loss += (labels[:,k].view(-1,1) * loss_matrix).sum() * (K-1) / len(labels)
    M1, M2 = K*(K-1)/2, K-1
    return pc_loss - M1 + M2

def ova_loss(f, labels, K):
    # labels = labels.max(dim=-1)[1]
    ova_loss = ((1-labels) * torch.log(1+torch.exp(-f))).sum(-1).mean() / (K-1)
    ova_loss += (labels * torch.log(1+torch.exp(f))).sum(-1).mean()
    # for k in range(K):
    #     ova_loss += (labels[:,k] * torch.log(1+torch.exp(f[:,k])) ).sum(-1).mean()
    # breakpoint()
    return ova_loss

def r_ova_loss(f, labels, p, K):
    # r_ova
    # cl = labels
    # ol = F.one_hot(torch.multinomial(1 - labels, 1), K).squeeze().float()

    # ova_loss = (ol * torch.log(1+torch.exp(-f))).sum(-1).mean()
    # ova_loss += (cl * torch.log(1+torch.exp(f))).sum(-1).mean()

    # return ova_loss

    # cost sensitive
    # f = -f

    # cost = labels
    # diff = f - cost
    # z = torch.where(cost == 0, 1, -1)

    # loss = torch.log(1 + torch.exp(z * diff)).sum(-1).mean()

    # return loss

    # normalization = (1-p) ** 2 / p + p
    # cost_pos = (1-p) / p / normalization
    # cost_neg = 1 / normalization

    # labels[labels == 0] = -1

    # cost = torch.where(labels > 0, cost_pos, cost_neg) 
    # loss = (cost * torch.log(1+torch.exp(-labels*f))).sum(-1).mean()

    # threashold = (1-p) / p / normalization

    # # return (loss - threashold).abs()
    # return F.relu(loss - threashold)

    # labels = 0.2 * labels + 0.8 * torch.ones_like(labels) / K
    logsm = F.log_softmax(f,1)
    # sm = F.softmax(-f,1)
    # loss = (labels * (-logsm/9+sm)).sum(-1).mean()
    loss = (labels * (-logsm)).sum(-1).mean()

    return (loss - 2.19722).abs()
