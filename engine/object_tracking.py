
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict
import cv2


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes in image space.
    Uses constant velocity model.
    """
    count = 0
    
    def __init__(self, bbox, class_id, score):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
            class_id: object class ID
            score: detection confidence
        """
        # Define constant velocity model
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Initialize state
        self.kf.statePost = self.bbox_to_state(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = class_id
        self.score = score
        
    def bbox_to_state(self, bbox):
        """Convert bbox [x1,y1,x2,y2] to state [x,y,s,r,vx,vy,vs]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2
        y = bbox[1] + h/2
        s = w * h
        r = w / float(h) if h != 0 else 1
        return np.array([[x], [y], [s], [r], [0], [0], [0]], dtype=np.float32)
    
    def state_to_bbox(self, state):
        """Convert state [x,y,s,r,vx,vy,vs] to bbox [x1,y1,x2,y2]"""
        w = np.sqrt(state[2] * state[3])
        h = state[2] / w if w != 0 else 1
        x1 = state[0] - w/2
        y1 = state[1] - h/2
        x2 = state[0] + w/2
        y2 = state[1] + h/2
        return np.array([x1, y1, x2, y2]).reshape((4,))
    
    def update(self, bbox, class_id, score):
        """Update the state with observed bbox."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.class_id = class_id
        self.score = score
        
        measurement = self.bbox_to_state(bbox)[:4]
        self.kf.correct(measurement)
    
    def predict(self):
        """Advance the state and return predicted bbox."""
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        bbox = self.state_to_bbox(self.kf.statePost.flatten())
        self.history.append(bbox)
        return bbox
    
    def get_state(self):
        """Return current bbox estimate."""
        return self.state_to_bbox(self.kf.statePost.flatten())


class OCSort:
    """
    OC-SORT: Observation-Centric SORT
    An improved version of SORT with better handling of occlusions.
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3):
        """
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: Minimum IOU for match
            delta_t: Time steps for velocity calculation
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.trackers = []
        self.frame_count = 0
    
    def update(self, bboxes, scores, class_ids, frame=None):
        """
        Update tracker with detections.
        
        Args:
            bboxes: List of bounding boxes [[x1,y1,x2,y2], ...]
            scores: List of confidence scores
            class_ids: List of class IDs
            frame: Current frame (optional, for visualization)
            
        Returns:
            List of tracked objects [[x1,y1,x2,y2,id,class_id,score], ...]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            bboxes, trks, self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(bboxes[m[0]], class_ids[m[0]], scores[m[0]])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(bboxes[i], class_ids[i], scores[i])
            self.trackers.append(trk)
        
        # Return tracked objects
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id, trk.class_id, trk.score])).reshape(1, -1))
            
            i -= 1
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))
    
    @staticmethod
    def iou_batch(bb_test, bb_gt):
        """
        Compute IOU between two sets of bboxes.
        
        Args:
            bb_test: (N, 4) array of [x1,y1,x2,y2]
            bb_gt: (M, 4) array of [x1,y1,x2,y2]
            
        Returns:
            (N, M) array of IOU scores
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return o
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assign detections to tracked objects using IOU.
        
        Returns:
            matched_indices, unmatched_detections, unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = self.iou_batch(np.array(detections), trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self.linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    @staticmethod
    def linear_assignment(cost_matrix):
        """Hungarian algorithm for linear assignment."""
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in x if i >= 0])
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))


class MultiObjectTracker:
    """
    Factory class for creating different types of trackers.
    """
    
    def __init__(self):
        """Initialize tracker factory."""
        self.tracker_type = None
    
    def ocsort(self, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3):
        """
        Create an OC-SORT tracker.
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: Minimum IOU for match
            delta_t: Time steps for velocity calculation
            
        Returns:
            OCSort tracker instance
        """
        self.tracker_type = "ocsort"
        return OCSort(max_age=max_age, min_hits=min_hits, 
                     iou_threshold=iou_threshold, delta_t=delta_t)
    
    def sort(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Create a SORT tracker (simplified version).
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: Minimum IOU for match
            
        Returns:
            OCSort tracker instance (using OC-SORT with default delta_t)
        """
        self.tracker_type = "sort"
        return OCSort(max_age=max_age, min_hits=min_hits, 
                     iou_threshold=iou_threshold, delta_t=1)