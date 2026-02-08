import cv2

# Function to handle mouse clicks
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: {x}, {y}")

# Read video and display
cap = cv2.VideoCapture('test_video.mp4')
ret, frame = cap.read()
cv2.imshow('Frame', frame)
cv2.setMouseCallback('Frame', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
