import cv2
import numpy as np
from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracker
from engine.aoi_utils import AOIAnalyzer
from database.mongo_database import MongoDatabase

# Initialize detector and tracker
od = ObjectDetection(model_path="models/yolov8n.pt")  
mot = MultiObjectTracker()
tracker = mot.ocsort(max_age=30, min_hits=3, iou_threshold=0.3)


# connect to MongoDB
mongo_db = MongoDatabase(uri="mongodb://localhost:27017/", db_name="object_tracking", collection_name="traffic_events")

vid_cap = cv2.VideoCapture("test_video.mp4")

# Get video properties
fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {width}x{height} @ {fps} FPS")

# Define Area of Interest (AOI) - can define multiple AOIs
aoi_polygons = [
    [[161, 178], [479, 146], [623, 199], [209, 272]],  # AOI 1
    # [[100, 100], [300, 100], [300, 300], [100, 300]]  # AOI 2 (example)
]
aoi_names = ["Restricted Zone"]  # , "Parking Area"]

# Initialize AOI Analyzer
aoi_analyzer = AOIAnalyzer(aoi_polygons, aoi_names)

# Optional: Save output video
output_path = "output_tracked_with_aoi.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = vid_cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Perform object detection
    bboxes, class_ids, scores = od.detect(frame, img_size=640, conf=0.25)
    
    # Update tracker
    tracked_objects = tracker.update(bboxes, scores, class_ids, frame)
    
    # Draw AOI polygons with statistics
    frame = aoi_analyzer.draw_aoi(frame, show_stats=True, show_names=True)
    
    # Process tracked objects and check AOI
    objects_in_aoi_count = 0
    
    for track in tracked_objects:
        x1, y1, x2, y2, obj_id, class_id, score = track.astype(int)
        bbox = [x1, y1, x2, y2]
        
        # Check if object is in AOI
        aoi_result = aoi_analyzer.check_object_in_aoi(bbox, obj_id, use_bottom_center=True)
        
        # Get class name and color
        class_name = od.get_class_name(class_id)
        color = od.get_color(class_id)
        
        # Change color if object is in AOI
        if aoi_result['in_aoi']:
            color = (0, 0, 255)  # Red for objects in AOI
            objects_in_aoi_count += 1
            record = {
                "object_id": int(obj_id),
                "object_name": class_name,
                "timestamp": cv2.getTickCount()
            }
            mongo_db.insert_record(record)

        
        # Draw bounding box
        thickness = 3 if aoi_result['in_aoi'] else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw tracking ID and class
        label = f"ID:{obj_id} {class_name}"
        if aoi_result['in_aoi']:
            label += f" [{', '.join(aoi_result['aoi_names'])}]"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw tracking point (bottom-center)
        tracking_point = aoi_result['point']
        cv2.circle(frame, tracking_point, 5, color, -1)
        
        # Draw entry/exit indicators
        if aoi_result['is_entry']:
            cv2.putText(frame, "ENTRY", (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif aoi_result['is_exit']:
            cv2.putText(frame, "EXIT", (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Display overall statistics
    stats_y = 30
    cv2.putText(frame, f"Frame: {frame_count} | Total Objects: {len(tracked_objects)} | In AOI: {objects_in_aoi_count}", 
               (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display AOI statistics
    all_stats = aoi_analyzer.get_statistics()
    for i, stats in all_stats.items():
        stats_y += 30
        stats_text = f"{stats['name']}: Current={stats['current']}, Entries={stats['entries']}, Exits={stats['exits']}"
        cv2.putText(frame, stats_text, (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Write frame to output video
    out.write(frame)
    
    # Display frame
    cv2.imshow("Object Tracking with AOI", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):  # Press 'r' to reset statistics
        aoi_analyzer.reset_statistics()
        print("Statistics reset!")

vid_cap.release()
out.release()
cv2.destroyAllWindows()

# Print final statistics
print("\n" + "="*50)
print("FINAL STATISTICS")
print("="*50)
final_stats = aoi_analyzer.get_statistics()
for i, stats in final_stats.items():
    print(f"\n{stats['name']}:")
    print(f"  Total Entries: {stats['entries']}")
    print(f"  Total Exits: {stats['exits']}")
    print(f"  Currently Inside: {stats['current']}")

print(f"\nOutput saved to: {output_path}")