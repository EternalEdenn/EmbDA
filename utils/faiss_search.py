import numpy as np
import faiss                   # make faiss available

def faiss_L2_search(xq, xb, k):
    '''
        xb: database
        xq: queries
        k: k nearest neighbors
    '''
    res = faiss.StandardGpuResources()  # use a single GPU
    index_flat = faiss.IndexFlatL2(xb.shape[1])   # build the index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(xb)         # add vectors to the index
    D, I = gpu_index_flat.search(xq, k)     # actual search

    return I

def faiss_cos_search(xq, xb, k):
    '''
        xb: database
        xq: queries
        k: k nearest neighbors
    '''

    res = faiss.StandardGpuResources()  # use a single GPU
    index_flat = faiss.IndexFlatIP(xb.shape[1])   # build the index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(xb)         # add vectors to the index
    D, I = gpu_index_flat.search(xq, k)     # actual search
    
    return I

