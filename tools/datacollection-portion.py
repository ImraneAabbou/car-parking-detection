import cv2
import pickle

# 1. Dimensions for the parking spots
width = 75
height = 150

# 2. Updated pickle file name
pickle_file = 'parking_positions_portion.pkl'

# Load existing positions if the file exists
try:
    with open(pickle_file, 'rb') as f:
        positionList = pickle.load(f)
except FileNotFoundError:
    positionList = []

def mouseclick(events, x, y, flags, params):
    global positionList
    
    # Left click to add a position
    if events == cv2.EVENT_LBUTTONDOWN:
        positionList.append((x, y))
        print(f"Added position: ({x}, {y})")
        
    # Right click to remove a position
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(positionList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                positionList.pop(i)
                print(f"Removed position: ({x1}, {y1})")
                break # Break to avoid modifying the list length during iteration
                
    # Save the updated list to the pickle file instantly
    with open(pickle_file, 'wb') as f:
        pickle.dump(positionList, f)

# Set up the window and mouse callback once outside the loop
cv2.namedWindow("Parking Selector")
cv2.setMouseCallback("Parking Selector", mouseclick)

while True:
    image = cv2.imread('./parking-portion.png')
    if image is None:
        print("Error: Could not find parking.png")
        break
        
    image = cv2.resize(image, (1920, 1080))
    
    # Draw the saved bounding boxes
    for pos in positionList:
        cv2.rectangle(image, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)
        
    cv2.imshow("Parking Selector", image)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
