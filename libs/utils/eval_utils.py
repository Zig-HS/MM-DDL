import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def segment_iou(target_segment, candidate_segments):
    t_start, t_end = target_segment
    c_starts = candidate_segments[:, 0]
    c_ends = candidate_segments[:, 1]

    inter_start = np.maximum(c_starts, t_start)
    inter_end = np.minimum(c_ends, t_end)
    inter_length = np.clip(inter_end - inter_start, 0, None)

    c_length = c_ends - c_starts
    t_length = t_end - t_start
    union_length = c_length + t_length - inter_length

    return inter_length / union_length

def evaluate_video(gt_segs, pred_segs, pred_scores, iou_threshold=0.5):
    if len(pred_segs) == 0:
        return 0, 0, len(gt_segs), [], []

    sorted_ind = np.argsort(pred_scores)[::-1]
    pred_segs = pred_segs[sorted_ind]
    pred_scores = pred_scores[sorted_ind]

    num_gt = len(gt_segs)
    num_pred = len(pred_segs)
    gt_matched = np.zeros(num_gt, dtype=bool)
    pred_results = []
    pred_scores_list = []

    for i in range(num_pred):
        pseg = pred_segs[i]
        if num_gt == 0:
            ious = np.array([])
        else:
            ious = segment_iou(pseg, gt_segs)
        max_iou = 0
        max_idx = -1
        if len(ious) > 0:
            max_iou = np.max(ious.cpu().numpy())
            max_idx = np.argmax(ious)

        if max_iou >= iou_threshold:
            if not gt_matched[max_idx]:
                gt_matched[max_idx] = True
                pred_results.append(1)
            else:
                pred_results.append(0)
        else:
            pred_results.append(0)
        pred_scores_list.append(pred_scores[i])

    tp = sum(pred_results)
    fp = num_pred - tp
    fn = num_gt - np.sum(gt_matched)
    return tp, fp, fn, pred_scores_list, pred_results

def dict_to_segs(data_dict):
    vidx = data_dict['video_id']
    fps = data_dict['fps']
    vlen = data_dict['duration']
    stride = data_dict['feat_stride']
    nframes = data_dict['feat_num_frames']
    segs = data_dict['segments'].detach().cpu()

    segs = (segs * stride + 0.5 * nframes) / fps
    segs[segs<=0.0] *= 0.0
    segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
    return segs
    # print("..")


class MetricCollector:
    def __init__(self, iou_thresholds=np.linspace(0.3, 0.7, 5)):
        self.all_labels = []  # 真实标签（0/1）
        self.all_pred_scores_auc = []  # AUC的预测得分

        # 按IoU阈值分组的AP计算
        self.iou_thresholds = iou_thresholds
        self.ap_true_labels = {iou: [] for iou in iou_thresholds}  # 每个IoU对应的TP/FP标签
        self.ap_pred_scores = {iou: [] for iou in iou_thresholds}  # 每个IoU对应的预测得分

        # 传统AR和Top-K AR
        self.recalls = []
        self.topk_list = [1, 5, 10]
        self.topk_recalls = {k: [] for k in self.topk_list}

    def update(self, data_dict, result):
        video_id = data_dict['video_id']
        is_forged = 1 if len(data_dict['segments']) > 0 else 0
        self.all_labels.append(is_forged)

        # AUC的预测得分
        if len(result['segments']) == 0:
            max_score = 0.0
        else:
            max_score = np.max(result['scores'])
        self.all_pred_scores_auc.append(max_score)

        # AP、AR 和 Top-K AR 的计算
        if is_forged:
            gt_segments = dict_to_segs(data_dict)
            pred_segments = result['segments']
            pred_scores = result['scores']

            # 按置信度排序
            sorted_ind = np.argsort(pred_scores)[::-1]
            pred_segments_sorted = pred_segments[sorted_ind]
            pred_scores_sorted = pred_scores[sorted_ind]

            # # 传统AR计算
            # tp, fp, fn, scores, labels = evaluate_video(gt_segments, pred_segments, pred_scores)
            # recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # self.recalls.append(recall)

            # Top-K AR计算
            for k in self.topk_list:
                pred_k_segments = pred_segments_sorted[:k]
                pred_k_scores = pred_scores_sorted[:k]
                tp_k, fp_k, fn_k, _, _ = evaluate_video(gt_segments, pred_k_segments, pred_k_scores)
                recall_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0
                self.topk_recalls[k].append(recall_k)

            # AP计算（按不同IoU阈值）
            for iou in self.iou_thresholds:
                _, _, _, scores_iou, labels_iou = evaluate_video(gt_segments, pred_segments, pred_scores, iou_threshold=iou)
                self.ap_true_labels[iou].extend(labels_iou)
                self.ap_pred_scores[iou].extend(scores_iou)

    def compute(self):
        auc_score = roc_auc_score(self.all_labels, self.all_pred_scores_auc) if self.all_labels else 0.0
        ap_scores = {}
        for iou in self.iou_thresholds:
            if len(self.ap_true_labels[iou]) > 0:
                ap = average_precision_score(self.ap_true_labels[iou], self.ap_pred_scores[iou])
                ap_scores[f"AP@{iou:.2f}"] = ap
            else:
                ap_scores[f"AP@{iou:.2f}"] = 0.0

        # mAP（所有IoU阈值的平均AP）
        mAP = np.mean(list(ap_scores.values())) if ap_scores else 0.0

        # 传统AR和Top-K AR
        # avg_recall = np.mean(self.recalls) if self.recalls else 0.0
        topk_ar = {}
        for k in self.topk_list:
            topk_ar[f"AR@{k}"] = np.mean(self.topk_recalls[k]) if self.topk_recalls[k] else 0.0
        topk_ar_avg = np.mean(list(topk_ar.values())) if topk_ar else 0.0

        return {
            "AUC": auc_score,
            "mAP": mAP,
            **ap_scores,  # 各IoU阈值下的AP
            "Top-K AR": topk_ar_avg,
            **topk_ar  # 每个k值的AR
        }


# class MetricCollector:
#     def __init__(self, topk_list=[1, 5, 10]):
#         self.all_labels = []  # 真实标签（0/1）
#         self.all_pred_scores_auc = []  # AUC的预测得分
#         self.ap_true_labels = []  # AP计算的真实标签（TP=1, FP=0）
#         self.ap_pred_scores = []  # AP计算的预测得分
#         self.recalls = []  # 每个伪造视频的Recall（传统AR）
#         self.topk_list = topk_list  # Top-K AR的K值列表
#         self.topk_recalls = {k: [] for k in topk_list}  # 每个k值下的Recall列表

#     def update(self, data_dict, result):
#         video_id = data_dict['video_id']
#         is_forged = 1 if len(data_dict['segments']) > 0 else 0
#         self.all_labels.append(is_forged)

#         # AUC的预测得分（最大置信度）
#         if len(result['segments']) == 0:
#             max_score = 0.0
#         else:
#             max_score = np.max(result['scores'])
#         self.all_pred_scores_auc.append(max_score)

#         # AP和传统AR的计算
#         if is_forged:
#             gt_segments = dict_to_segs(data_dict)
#             pred_segments = result['segments']
#             pred_scores = result['scores']

#             # 按置信度排序
#             sorted_ind = np.argsort(pred_scores)[::-1]
#             pred_segments_sorted = pred_segments[sorted_ind]
#             pred_scores_sorted = pred_scores[sorted_ind]

#             # 传统AR计算（所有预测）
#             tp, fp, fn, scores, labels = evaluate_video(gt_segments, pred_segments, pred_scores)
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#             self.recalls.append(recall)

#             # Top-K AR计算
#             for k in self.topk_list:
#                 pred_k_segments = pred_segments_sorted[:k]
#                 pred_k_scores = pred_scores_sorted[:k]
#                 tp_k, fp_k, fn_k, _, _ = evaluate_video(gt_segments, pred_k_segments, pred_k_scores)
#                 recall_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0
#                 self.topk_recalls[k].append(recall_k)

#     def compute(self):
#         auc_score = roc_auc_score(self.all_labels, self.all_pred_scores_auc) if self.all_labels else 0.0
#         ap_score = average_precision_score(self.ap_true_labels, self.ap_pred_scores) if self.ap_true_labels else 0.0
#         avg_recall = np.mean(self.recalls) if self.recalls else 0.0

#         # Top-K AR 计算
#         topk_ar = {}
#         for k in self.topk_list:
#             topk_ar[f"AR@{k}"] = np.mean(self.topk_recalls[k]) if self.topk_recalls[k] else 0.0

#         # Top-K AR 的平均值
#         topk_ar_avg = np.mean(list(topk_ar.values())) if topk_ar else 0.0

#         return {
#             "AUC": auc_score,
#             "AP": ap_score,
#             "AR": avg_recall,  # 传统AR（所有预测）
#             "Top-K AR": topk_ar_avg,  # 所有k的Recall平均
#             **topk_ar  # 每个k的Recall
#         }

# class MetricCollector:
#     def __init__(self):
#         self.all_labels = []  # 真实标签（0/1）
#         self.all_pred_scores_auc = []  # AUC的预测得分
#         self.ap_true_labels = []  # AP计算的真实标签（TP=1, FP=0）
#         self.ap_pred_scores = []  # AP计算的预测得分
#         self.recalls = []  # 每个伪造视频的Recall

#     def update(self, data_dict, result):
#         video_id = data_dict['video_id']
#         is_forged = 1 if len(data_dict['segments']) > 0 else 0
#         self.all_labels.append(is_forged)

#         # AUC的预测得分
#         if len(result['segments']) == 0:
#             max_score = 0.0
#         else:
#             max_score = np.max(result['scores'])
#         self.all_pred_scores_auc.append(max_score)

#         # AP和AR的计算
#         if is_forged:
#             gt_segments = dict_to_segs(data_dict)#['segments']
#             pred_segments = result['segments']
#             pred_scores = result['scores']
#             tp, fp, fn, scores, labels = evaluate_video(gt_segments, pred_segments, pred_scores)
#             self.ap_true_labels.extend(labels)
#             self.ap_pred_scores.extend(scores)
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#             self.recalls.append(recall)

#     def compute(self):
#         auc_score = roc_auc_score(self.all_labels, self.all_pred_scores_auc)
#         ap_score = average_precision_score(self.ap_true_labels, self.ap_pred_scores) if self.ap_true_labels else 0.0
#         avg_recall = np.mean(self.recalls) if self.recalls else 0.0
#         return {
#             "AUC": auc_score,
#             "AP": ap_score,
#             "AR": avg_recall
#         }