import os
import numpy as np
import ot
from ot.backend import get_backend
import torch.nn.functional as F
import torch
from typing import List, Tuple

'Part for OT'
def max_scaling(C):
    eps = 1e-10
    # Min-max scaling for stabilization
    nx = get_backend(C)
    C_max = nx.max(C)
    C = (C + eps) / (C_max + eps)
    return C

def cmpt_dist_matrix(d1_embs: torch.Tensor, d2_embs: torch.Tensor,
                     dist: str='cos'
)-> torch.Tensor:
    if dist == 'l2':
        C = torch.cdist(d1_embs, d2_embs, p=2)
    elif dist == 'cos':
        C = torch.matmul(d1_embs, d2_embs.t())  # Range 0-1
        C = 1.0 - C  # Convert to distance
    C = max_scaling(C)  # Range 0-1
    return C

def cal_ot(d1_embs: torch.Tensor, d2_embs: torch.Tensor,
           d1_weights: List[torch.Tensor], d2_weights: List[torch.Tensor], 
           dist: str='cos'
)-> torch.Tensor:
    """
    OT: Optimal Transport or SMD: Sentence Movers' Distance

    Parameters:
        d1_embs_1, d2_embs_2 : torch.Tensor, shape = [n, dim], [m, dim]
            Embeddings of Source / Target Document
        d1_weights, d2_weights : torch.Tensor, shape = [n], [m]
            Weights of Embeddings in Source / Target Document
        dist : str 
            "cos" | "l2"
    Return:
        torch.tensor: OT Distance
    Resource: 
        This code partly refers to OTAlign.
        https://github.com/yukiar/OTAlign
    """

    C = cmpt_dist_matrix(d1_embs, d2_embs, dist)
    OT = ot.sinkhorn2(d1_weights, d2_weights, C, reg=0.1, stopThr=1e-04,
                            numItermax=1000)
    return(OT)


'Part for BiMax'
def bimax_loop(seg_embs_1: torch.Tensor, 
               seg_embs_2: torch.Tensor
)-> torch.Tensor:
    """
    BiMax : Bi-directional Maxsim Score

    Parameters:
        seg_embs_1, seg_embs_2 : torch.Tensor, shape = [n, dim], [m, dims]
            Embeddings of Source / Target Document
    Return:
        torch.Tensor: BiMax Similarity Score
    """

    similarity = seg_embs_1 @ seg_embs_2.T
    row_max = torch.max(similarity, dim=1).values
    col_max = torch.max(similarity, dim=0).values
    score_1 = row_max.mean()
    score_2 = col_max.mean()
    score = ( score_1 + score_2 )/2
    return score

def pad_stack(tensors: List[torch.Tensor], dim: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters:
        tensors : List[torch.Tensor]:
            A list of length P, each of shape [len_p, dim]
    Return:
        padded : [P, Lmax, dim]
        mask :   [P, Lmax]
    """
    lengths = torch.tensor([t.shape[0] for t in tensors], device=device)
    Lmax = int(lengths.max().item())
    P = len(tensors)
    if Lmax == 0:
        return (torch.empty(P, 0, dim, device=device, dtype=tensors[0].dtype),
                torch.empty(P, 0, device=device, dtype=torch.bool))
    padded = torch.zeros(P, Lmax, dim, device=device, dtype=tensors[0].dtype)
    mask = torch.zeros(P, Lmax, device=device, dtype=torch.bool)
    for p, t in enumerate(tensors):
        lp = t.shape[0]
        if lp > 0:
            padded[p, :lp] = t
            mask[p, :lp] = True
    return padded, mask

def bimax_batch(
    src_seg_embs: List[torch.Tensor],     # len=N， [ns_i, dim]
    tgt_seg_embs: List[torch.Tensor],     # len=M， [nt_j, dim]
    I: torch.Tensor,                      # [N, faiss_k]，candidates' indices from faiss search
    batch_pairs: int = 2048,              # number of pairs in a batch
    col_chunk: int = 2048,                # The sub-block size of the similarity matrix in the "column" (target segment) direction
    device: str = "cpu"
) -> torch.Tensor:
    """
    Parameters:
        src_seg_embs, tgt_seg_embs : List[torch.Tensor]
            Two lists contain all the source/target document embeddings
        I : torch.Tensor
            Result matrix of faiss search
        batch_pairs : int
            number of pairs in a batch
        col_chunk : int
            sub-block size of the similarity matrix in the "column" direction
        device : str
            "cpu" or "cuda"
    Return:
        torch.Tensor: BiMax scores for all the pairs, shape=[N, faiss_k]
    """
    if not isinstance(I, torch.Tensor):
        I = torch.as_tensor(I, device=device)

    N, faiss_k = I.shape
    dim = src_seg_embs[0].shape[-1]

    src_ids = torch.arange(N, device=device).repeat_interleave(faiss_k)            # [N*faiss_k]
    tgt_ids = I.reshape(-1)                                                        # [N*faiss_k]
    num_pairs = src_ids.numel()

    out = torch.empty(num_pairs, device=device, dtype=src_seg_embs[0].dtype)

    finfo_min = torch.finfo(src_seg_embs[0].dtype).min 

    for start in range(0, num_pairs, batch_pairs):
        end = min(start + batch_pairs, num_pairs)
        batch_src_ids = src_ids[start:end].tolist()
        batch_tgt_ids = tgt_ids[start:end].tolist()

        src_list = [src_seg_embs[i] for i in batch_src_ids]  
        tgt_list = [tgt_seg_embs[j] for j in batch_tgt_ids]   

        S, S_mask = pad_stack(src_list, dim, device)  # [P, Smax, dim], [P, Smax]
        T, T_mask = pad_stack(tgt_list, dim, device)  # [P, Tmax, dim], [P, Tmax]
        P, Smax, _ = S.shape
        _, Tmax, _ = T.shape

        if Smax == 0 or Tmax == 0:
            out[start:end] = 0.0
            continue

        row_max = torch.full((P, Smax), finfo_min, device=device, dtype=S.dtype)
        col_max = torch.full((P, Tmax), finfo_min, device=device, dtype=S.dtype)

        for c0 in range(0, Tmax, col_chunk):
            c1 = min(c0 + col_chunk, Tmax)
            T_blk = T[:, c0:c1, :]                     # [P, C, dim]
            T_blk_mask = T_mask[:, c0:c1]              # [P, C]

            sim_blk = torch.einsum('psd,pdc->psc', S, T_blk.transpose(1, 2))

            sim_blk = sim_blk.masked_fill(~S_mask.unsqueeze(-1), finfo_min)
            sim_blk = sim_blk.masked_fill(~T_blk_mask.unsqueeze(1), finfo_min)

            row_max_blk = sim_blk.amax(dim=-1)     # [P, Smax]
            row_max = torch.maximum(row_max, row_max_blk)

            col_max_blk = sim_blk.amax(dim=-2)     # [P, C]
            col_max[:, c0:c1] = torch.maximum(col_max[:, c0:c1], col_max_blk)

            del sim_blk, row_max_blk, col_max_blk

        row_sum = (row_max.masked_fill(~S_mask, 0.0)).sum(dim=1)         # [P]
        row_cnt = S_mask.sum(dim=1).clamp_min(1)                         # [P]
        score_1 = row_sum / row_cnt

        col_sum = (col_max.masked_fill(~T_mask, 0.0)).sum(dim=1)         # [P]
        col_cnt = T_mask.sum(dim=1).clamp_min(1)                         # [P]
        score_2 = col_sum / col_cnt

        out[start:end] = 0.5 * (score_1 + score_2)

        del S, S_mask, T, T_mask, row_max, col_max

    return out.view(N, faiss_k)         # [N, faiss_k]


'Part for GMD'
def GMD(docVecA: torch.Tensor, docVecB: torch.Tensor, 
        weightsA: list, weightsB: list, 
        metric:str = "l2", device: str = "cpu"):
    """
    GMD: Greedy Movers' Distance

    Parameters:
        docVecA, docVecB : torch.Tensor, shape = [n, dim]
            Embeddings of doc A and doc B
        weightsA, weightsB : torch.Tensor, shape = [n]
            Weights of Embeddings in Doc A / Doc B
        metric : str
            "l2" | "cos" 
        device : str
            "cpu" or "cuda"
    Return
        torch.tensor: GMD Distance
    Resource:
        This code partly refers to the GitHub 
        https://github.com/nlpcuom/parallel_corpus_mining/blob/master/document_alignment/GreedyMoversDistance.py,
        but it was adapted to run on a Torch-based implementation,
        and the algorithm was optimized.
    """
    try:
        sortedPairs = getSortedDistances(docVecA, docVecB, metric)

        distance = torch.tensor(0.0, device=device, dtype=torch.float32)
        for pair in sortedPairs:
            weigVecA = weightsA[pair["i"]]
            weigVecB = weightsB[pair["j"]]
            flow = torch.minimum(weigVecA, weigVecB)
            weightsA[pair["i"]] = weigVecA - flow
            weightsB[pair["j"]] = weigVecB - flow
            vecA = docVecA[pair["i"]]
            vecB = docVecB[pair["j"]]

            if metric == "cos":
                dist = 1 - torch.dot(vecA, vecB) / (torch.norm(vecA) * torch.norm(vecB))
                distance += dist * flow
            elif metric == "l2":
                dist = torch.norm(vecA - vecB)
                distance += dist * flow
            else:
                raise ValueError("Invalid metric")
        return distance
    except Exception as e:
        print("Error in GreedyMoversDistance:", e)
        return 0.0

def getSortedDistances( docVecA: torch.Tensor,
                        docVecB: torch.Tensor,
                        metric: str = "l2"):
    if metric == "l2":
        D = torch.cdist(docVecA, docVecB, p=2)
    elif metric == "cos":
        S = docVecA @ docVecB.T        
        D = 1 - S         
    else:
        raise ValueError("Invalid metric")

    m, n = D.shape
    flat = D.reshape(-1) 

    vals, idx = torch.sort(flat, descending=False)

    i_idx = (idx // n).to(torch.int64)
    j_idx = (idx %  n).to(torch.int64)

    result = [{"dist": float(vals[k]), "i": int(i_idx[k]), "j": int(j_idx[k])}
              for k in range(vals.numel())]

    return result