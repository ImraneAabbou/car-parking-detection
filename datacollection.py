import cv2
import pickle
import os

# 1. ADJUSTED DIMENSIONS for smaller parking spots
# You might need to tweak these slightly (e.g., 30x60) 
width = 40 
height = 65

# 2. SEPARATE DIRECTORY for the new parking lot
save_dir = 'cropped_img'
pickle_file = 'car_position_parking.pkl'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_cropped_img(img, pos, index):
    cropped_img = img[pos[1]:pos[1]+height, pos[0]:pos[0]+width]
    save_path = os.path.join(save_dir, f'pos_{index}.png')
    cv2.imwrite(save_path, cropped_img)
    print(f'saved cropped image: {save_path}')

try:
    with open(pickle_file, 'rb') as f:
        positionList = pickle.load(f)
except:
    positionList = []

def mouseclick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        positionList.append((x, y))
        # 3. CHANGED FILENAME TO 'parking.png'
        img_full = cv2.imread('parking.png')
        img_resized = cv2.resize(img_full, (1280, 720))
        save_cropped_img(img_resized, (x, y), len(positionList))
        
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(positionList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                positionList.pop(i)
                
    with open(pickle_file, 'wb') as f:
        pickle.dump(positionList, f)

while True:
    # 4. LOAD THE SECOND IMAGE
    image = cv2.imread('parking.png')
    if image is None:
        print("Error: Could not find parking.png")
        break
        
    image = cv2.resize(image, (1920, 1080))
    
    for pos in positionList:
        cv2.rectangle(image, pos, (pos[0]+width, pos[1]+height), (255, 0, 255), 2)
        
    cv2.imshow("Parking Selector", image)
    cv2.setMouseCallback("Parking Selector", mouseclick)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
