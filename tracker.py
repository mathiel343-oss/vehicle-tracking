import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.hits = 1
        self.no_losses = 0

    def is_confirmed(self):
        return self.hits >= 1

    def to_ltrb(self):
        x, y, w, h = self.bbox
        return x, y, x + w, y + h


class DeepSort:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0

    def update_tracks(self, detections, frame=None):
        det_boxes = [det[0] for det in detections]
        
        # إذا لم تكن هناك مسارات سابقة، أضف كافة الاكتشافات كمسارات جديدة
        if len(self.tracks) == 0:
            for box in det_boxes:
                self.tracks.append(Track(self.track_id_count, box))
                self.track_id_count += 1
            return self.tracks

        # 1. بناء مصفوفة تكلفة IoU (Cost Matrix)
        num_tracks = len(self.tracks)
        num_dets = len(det_boxes)
        
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det_box in enumerate(det_boxes):
                iou_matrix[t_idx, d_idx] = self._iou(track.bbox, det_box)

        # 2. المطابقة المثالية باستخدام Hungarian Algorithm
        # نحول الـ IoU إلى Cost بضربها في 1-
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)

        matched_track_indices = set()
        matched_det_indices = set()

        for r, c in zip(row_indices, col_indices):
            # التأكد من أن نسبة التقاطع أكبر من الحد الأدنى
            if iou_matrix[r, c] >= self.iou_threshold:
                self.tracks[r].bbox = det_boxes[c]
                self.tracks[r].hits += 1
                self.tracks[r].no_losses = 0
                
                matched_track_indices.add(r)
                matched_det_indices.add(c)

        # 3. معالجة المسارات غير المطابقة (Unmatched Tracks)
        updated_tracks = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx in matched_track_indices:
                updated_tracks.append(track)
            else:
                track.no_losses += 1
                # إبقاء المسار لفترة max_age قبل حذفه
                if track.no_losses < self.max_age:
                    updated_tracks.append(track)

        # 4. إضافة الاكتشافات الجديدة (Unmatched Detections)
        for d_idx, det_box in enumerate(det_boxes):
            if d_idx not in matched_det_indices:
                updated_tracks.append(Track(self.track_id_count, det_box))
                self.track_id_count += 1

        self.tracks = updated_tracks
        return self.tracks

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)