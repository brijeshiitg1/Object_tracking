
import cv2
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


class AOIAnalyzer:
    """
    Area of Interest (AOI) analyzer for tracking objects within defined polygons.
    """
    
    def __init__(self, aoi_polygons: List[List[List[int]]], aoi_names: List[str] = None):
        """
        Initialize AOI analyzer.
        
        Args:
            aoi_polygons: List of polygons, each polygon is a list of points [[x,y], [x,y], ...]
            aoi_names: Optional names for each AOI region
        """
        self.aoi_polygons = [np.array(poly, dtype=np.int32) for poly in aoi_polygons]
        self.aoi_names = aoi_names or [f"AOI_{i+1}" for i in range(len(aoi_polygons))]
        
        # Track objects in each AOI
        self.objects_in_aoi = defaultdict(set)  # {aoi_index: {obj_ids}}
        self.object_aoi_history = defaultdict(list)  # {obj_id: [aoi_indices]}
        
        # Statistics
        self.entry_count = defaultdict(int)  # Count entries per AOI
        self.exit_count = defaultdict(int)   # Count exits per AOI
        self.current_count = defaultdict(int)  # Current objects in AOI
    
    def point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """
        Check if a point is inside a polygon using cv2.pointPolygonTest.
        
        Args:
            point: (x, y) coordinates
            polygon: Numpy array of polygon points
            
        Returns:
            True if point is inside polygon
        """
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0
    
    def bbox_center(self, bbox: List[float]) -> Tuple[int, int]:
        """
        Calculate center point of bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            (cx, cy) center coordinates
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy
    
    def bbox_bottom_center(self, bbox: List[float]) -> Tuple[int, int]:
        """
        Calculate bottom-center point of bounding box (often more accurate for tracking).
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            (cx, bottom_y) coordinates
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        bottom_y = int(y2)
        return cx, bottom_y
    
    def check_object_in_aoi(self, bbox: List[float], obj_id: int, 
                           use_bottom_center: bool = True) -> Dict:
        """
        Check if an object is in any AOI and update statistics.
        
        Args:
            bbox: [x1, y1, x2, y2]
            obj_id: Object tracking ID
            use_bottom_center: Use bottom-center point instead of center
            
        Returns:
            Dict with AOI information: {
                'in_aoi': bool,
                'aoi_indices': List[int],
                'aoi_names': List[str],
                'is_entry': bool,
                'is_exit': bool
            }
        """
        # Get reference point
        if use_bottom_center:
            point = self.bbox_bottom_center(bbox)
        else:
            point = self.bbox_center(bbox)
        
        current_aois = set()
        aoi_names = []
        
        # Check which AOIs contain the object
        for i, polygon in enumerate(self.aoi_polygons):
            if self.point_in_polygon(point, polygon):
                current_aois.add(i)
                aoi_names.append(self.aoi_names[i])
        
        # Get previous AOIs for this object
        previous_aois = self.objects_in_aoi.get(obj_id, set())
        
        # Detect entry/exit
        is_entry = bool(current_aois - previous_aois)  # New AOIs
        is_exit = bool(previous_aois - current_aois)   # Left AOIs
        
        # Update statistics
        if is_entry:
            for aoi_idx in (current_aois - previous_aois):
                self.entry_count[aoi_idx] += 1
                self.current_count[aoi_idx] += 1
        
        if is_exit:
            for aoi_idx in (previous_aois - current_aois):
                self.exit_count[aoi_idx] += 1
                self.current_count[aoi_idx] -= 1
        
        # Update tracking
        self.objects_in_aoi[obj_id] = current_aois
        self.object_aoi_history[obj_id].append(list(current_aois))
        
        return {
            'in_aoi': bool(current_aois),
            'aoi_indices': list(current_aois),
            'aoi_names': aoi_names,
            'point': point,
            'is_entry': is_entry,
            'is_exit': is_exit
        }
    
    def get_statistics(self, aoi_index: int = None) -> Dict:
        """
        Get statistics for an AOI or all AOIs.
        
        Args:
            aoi_index: Specific AOI index, or None for all AOIs
            
        Returns:
            Dict with statistics
        """
        if aoi_index is not None:
            return {
                'name': self.aoi_names[aoi_index],
                'entries': self.entry_count[aoi_index],
                'exits': self.exit_count[aoi_index],
                'current': self.current_count[aoi_index]
            }
        else:
            return {
                i: {
                    'name': self.aoi_names[i],
                    'entries': self.entry_count[i],
                    'exits': self.exit_count[i],
                    'current': self.current_count[i]
                }
                for i in range(len(self.aoi_polygons))
            }
    
    def draw_aoi(self, frame: np.ndarray, show_stats: bool = True, 
                 show_names: bool = True) -> np.ndarray:
        """
        Draw AOI polygons and statistics on frame.
        
        Args:
            frame: Input frame
            show_stats: Whether to show statistics
            show_names: Whether to show AOI names
            
        Returns:
            Frame with AOI visualization
        """
        for i, polygon in enumerate(self.aoi_polygons):
            # Draw polygon
            color = (0, 255, 255)  # Yellow
            cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
            
            # Fill with semi-transparent color
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            
            # Get top-left point for text
            text_pos = tuple(polygon[0])
            
            # Draw AOI name
            if show_names:
                cv2.putText(frame, self.aoi_names[i], 
                           (text_pos[0], text_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw statistics
            if show_stats:
                stats = self.get_statistics(i)
                stats_text = f"In: {stats['current']} | Total: {stats['entries']}"
                cv2.putText(frame, stats_text,
                           (text_pos[0], text_pos[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.objects_in_aoi.clear()
        self.object_aoi_history.clear()
        self.entry_count.clear()
        self.exit_count.clear()
        self.current_count.clear()