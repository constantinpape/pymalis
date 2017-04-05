from libcpp.vector cimport vector
from libc.stdint cimport int64_t
from libcpp cimport bool
import numpy as np
cimport numpy as np

def malis(affs, gt):

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not affs.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous affinity arrray (avoid this by passing C_CONTIGUOUS arrays)")
        affs = np.ascontiguousarray(affs)
    if gt is not None and not gt.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous ground-truth arrray (avoid this by passing C_CONTIGUOUS arrays)")
        gt = np.ascontiguousarray(gt)

    print("Preparing loss volumes...")

    loss_pos = np.zeros(affs.shape, dtype=np.float32)
    loss_neg = np.zeros(affs.shape, dtype=np.float32)

    __malis(affs, gt, loss_pos, loss_neg)

    return (loss_pos, loss_neg)

def __malis(
        np.ndarray[np.float32_t, ndim=4] affs,
        np.ndarray[np.int64_t, ndim=3]   gt,
        np.ndarray[np.float32_t, ndim=4] loss_pos,
        np.ndarray[np.float32_t, ndim=4] loss_neg):

    cdef float*   aff_data
    cdef int64_t* gt_data
    cdef float*   loss_pos_data
    cdef float*   loss_neg_data
    aff_data = &affs[0,0,0,0]
    gt_data = &gt[0,0,0]
    loss_pos_data = &loss_pos[0,0,0,0]
    loss_neg_data = &loss_neg[0,0,0,0]

    ___malis(
            affs.shape[1], affs.shape[2], affs.shape[3],
            aff_data,
            gt_data,
            loss_pos_data,
            loss_neg_data)

cdef extern from "c_frontend.h":

    void ___malis(
            size_t         depth,
            size_t         height,
            size_t         width,
            const float*   affinity_data,
            const int64_t* groundtruth_data,
            float*         loss_pos_data,
            float*         loss_neg_data);
