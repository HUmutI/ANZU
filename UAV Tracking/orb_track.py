# ORB Algorithm for Object Tracking

import cv2
import matplotlib.pyplot as plt 
import numpy as np
import time

# this is target image 
image = cv2.imread("./photos/uav_zoom_2.png")
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
rgb_image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)

# Initiate ORB
orb = cv2.ORB_create()

# find the keypoints with ORB
keypoints_1, descriptors_1 = orb.detectAndCompute(gray_image, None)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(rgb_image,keypoints_1,None,color=(0,255,0), flags=0)

plt.imshow(img2)

# path to video  
video_path="./videos/Drone Chasing edited.mp4"  
video = cv2.VideoCapture(video_path)

# Initialize variables for FPS calculation
t0 = time.time()
n_frames = 1

# Initiate
orb = cv2.ORB_create()

# matcher object
bf = cv2.BFMatcher()


while True :
# reading video 
    ret,frame=video.read()

    if ret:
          # convert frame to gray scale 
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
        
        # compute the descriptors with BRIEF
        keypoints_2, descriptors_2 =  orb.detectAndCompute(frame_gray, None)
        
        """
        Compare the keypoints/descriptors extracted from the 
        first frame(from target object) with those extracted from the current frame.
        """
        matches =bf.match(descriptors_1, descriptors_2)
    
    
        for match in matches:
        
            # queryIdx gives keypoint index from target image
            query_idx = match.queryIdx
            
            # .trainIdx gives keypoint index from current frame 
            train_idx = match.trainIdx
            
            # take coordinates that matches
            pt1 = keypoints_1[query_idx].pt
            
            # current frame keypoints coordinates
            pt2 = keypoints_2[train_idx].pt
            
            # draw circle to pt2 coordinates , because pt2 gives current frame coordinates
            cv2.circle(frame,(int(pt2[0]),int(pt2[1])),2,(255,0,0),2)
    
        elapsed_time = time.time() - t0
        avg_fps = (n_frames / elapsed_time)
        print("Average FPS: " + str(avg_fps))
        cv2.putText(frame, str(avg_fps) , (50,50) , cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 1, cv2.LINE_AA)
        n_frames += 1
    
        #cv2.putText(frame,f"FPS :{str(avg_fps)}" , (50,50) , cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0), 2, cv2.LINE_AA)
    
        cv2.imshow("coordinate_screen",frame) 
    
    
        k = cv2.waitKey(5) & 0xFF # after drawing rectangle press esc   
        if k == 27:
            cv2.destroyAllWindows()
            break
    else:
        break
  
cv2.destroyAllWindows()