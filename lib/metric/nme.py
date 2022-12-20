import torch
import numpy as np


class NME:
    def __init__(self, nme_left_index, nme_right_index):
        self.nme_left_index = nme_left_index
        self.nme_right_index = nme_right_index

    def __repr__(self):
        return "NME()"

    def test(self, label_pd, label_gt):
        sum_nme = 0
        total_cnt = 0
        label_pd = label_pd.data.cpu().numpy()
        label_gt = label_gt.data.cpu().numpy()
        for i in range(label_gt.shape[0]):
            landmarks_gt = label_gt[i]
            landmarks_pv = label_pd[i]
            pupil_distance = np.linalg.norm(landmarks_gt[self.nme_left_index] - landmarks_gt[self.nme_right_index])
            landmarks_delta = landmarks_pv - landmarks_gt
            nme = (np.linalg.norm(landmarks_delta, axis=1) / pupil_distance).mean()
            sum_nme += nme
            total_cnt += 1
        return sum_nme, total_cnt
