import torch
oc_maxmin_dict = {}
ic_maxmin_dict = {}
import numpy as np

def R1_reorder(layer_norm, qproj, kproj, vproj, index, counts):
    layer_norm.register_buffer("reorder_index", index)                                                                  # layer_norm的weight和bias需要先进行reorder, 就相当于对qkv的输入进行了reorder
    layer_norm.weight.data = torch.index_select(layer_norm.weight.data, 0, index)                                       # 增加reorder部分！
    layer_norm.bias.data = torch.index_select(layer_norm.bias.data, 0, index)

    qproj.act_quantizer.cluster_dim = 2
    qproj.act_quantizer.cluster_counts = counts                                                                         # 对qkv的输入进行聚类划分
    kproj.act_quantizer.cluster_dim = 2
    kproj.act_quantizer.cluster_counts = counts
    vproj.act_quantizer.cluster_dim = 2
    vproj.act_quantizer.cluster_counts = counts

    qproj.weight.data = torch.index_select(qproj.weight.data, 1, index)                                                 # 对qkv矩阵的权重进行一个reorder
    qproj.set_ic_cluster_counts(counts, a_dim=2)
    kproj.weight.data = torch.index_select(kproj.weight.data, 1, index)                                              
    kproj.set_ic_cluster_counts(counts, a_dim=2)
    vproj.weight.data = torch.index_select(vproj.weight.data, 1, index)                                                 
    vproj.set_ic_cluster_counts(counts, a_dim=2)


def R2_reorder(qproj, kproj, qkt_matmul, index, counts):
    qproj.weight.data = torch.index_select(qproj.weight.data, 0, index)
    qproj.bias.data = torch.index_select(qproj.bias.data, 0, index)
    kproj.weight.data = torch.index_select(kproj.weight.data, 0, index)
    kproj.bias.data = torch.index_select(kproj.bias.data, 0, index)
    qkt_matmul.set_ic_cluster_counts(counts, x1_dim=2, x2_dim=2)


def R3_reorder(vproj, pv_matmul, out_proj, index, counts):
    vproj.weight.data = torch.index_select(vproj.weight.data, 0, index)
    vproj.bias.data = torch.index_select(vproj.bias.data, 0, index)
    pv_matmul.set_ic_cluster_counts(counts, cluster_x1=False)
    out_proj.weight.data = torch.index_select(out_proj.weight.data, 1, index)
    out_proj.set_ic_cluster_counts(counts)


def R4_reorder(layer_norm, fc1, index, counts):
    layer_norm.register_buffer("reorder_index", index)
    layer_norm.weight.data = torch.index_select(layer_norm.weight.data, 0, index)                                       # 增加reorder部分！
    layer_norm.bias.data = torch.index_select(layer_norm.bias.data, 0, index)

    fc1.act_quantizer.cluster_dim = 1                                                                                   # 因为这时候的X都是2D的了
    fc1.act_quantizer.cluster_counts = counts

    fc1.weight.data = torch.index_select(fc1.weight.data, 1, index)
    fc1.set_ic_cluster_counts(counts, a_dim=1)


def R5_reorder(fc1, fc2, index, counts):
    fc1.weight.data = torch.index_select(fc1.weight.data, 0, index)
    fc1.bias.data = torch.index_select(fc1.bias.data, 0, index)
    fc2.weight.data = torch.index_select(fc2.weight.data, 1, index)
    fc2.set_ic_cluster_counts(counts, a_dim=1)


"""
ltl.
This hook funciton.
For R1, R2, R4 reorder.
It computes and record the maximum and minimum value in every output channel of one layer.
The output could be a 2D or 3D tensor.
"""
def layer_omax_hook(m, i, o):
    name = m.name
    if not isinstance(o, torch.Tensor):
        return
    if o.ndim == 3:
        xmax = torch.amax(o, [0, 1])  # shape d
        xmin = torch.amin(o, [0, 1])  # shape d
    elif o.ndim == 2:
        xmax = torch.amax(o, [0])  # shape d
        xmin = torch.amin(o, [0])  # shape d

    if name not in oc_maxmin_dict:
        oc_maxmin_dict[name] = (xmax.detach_(), xmin.detach_())
    else:
        oc_maxmin_dict[name] = (
            torch.max(oc_maxmin_dict[name][0], xmax).detach_(),
            torch.min(oc_maxmin_dict[name][1], xmin).detach_(),
        )


"""
ltl
For R3, R5 reorder.
"""
def layer_i0max_hook(m, i, o):
    name = m.name
    if len(i) == 0 or not isinstance(i[0], torch.Tensor):
        return
    if i[0].ndim == 3:
        xmax = torch.amax(i[0], [0, 1])  # shape d
        xmin = torch.amin(i[0], [0, 1])  # shape d
    elif i[0].ndim == 2:
        xmax = torch.amax(i[0], [0])  # shape d
        xmin = torch.amin(i[0], [0])  # shape d

    if name not in ic_maxmin_dict:
        ic_maxmin_dict[name] = xmax.detach_(), xmin.detach_()
    else:
        ic_maxmin_dict[name] = (
            torch.max(ic_maxmin_dict[name][0], xmax).detach_(),
            torch.min(ic_maxmin_dict[name][1], xmin).detach_(),
        )

def peg_tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None):
    """
    x shape [b,n,d]
    paper: Understanding and Overcoming the Challenges of Efficient Transformer Quantization
    """
    if n_heads is None:
        n_heads = 1

    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1)
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1)
        npdatamax = xmax.view(n_heads, -1, n)
        npdatamin = xmin.view(n_heads, -1, n)
        npdata = (npdatamax[:,:,0]-npdatamin[:,:,0]).reshape(n_heads,-1)
    else:
        npdatamax = xmax.view(n_heads, -1, 1)
        npdatamin = xmin.view(n_heads, -1, 1)
    # npdata = np.concatenate([npdatamax, npdatamin], -1)

        npdata = (npdatamax-npdatamin).reshape(n_heads,-1)

    cnt = 0
    all_index = []
    all_counts = []
    for i, data in enumerate(npdata):
        # for each head
        index=torch.argsort(data)
        counts=[len(data)//n_clusters]*n_clusters
        index += cnt
        all_index.append(index)
        # breakpoint()
        all_counts.append(np.array(counts))
        cnt += len(data)
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts

tensor_calc_reorder_index=peg_tensor_calc_reorder_index