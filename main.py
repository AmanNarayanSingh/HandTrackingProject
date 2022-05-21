import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

#handDetection module object declaration (module created so that we can simply use this module in future projects without re writing the same code)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils   #to draw lines between 21 landmarks
pTime=0             #previous time
cTime=0             #current time

while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #hands on line 9 only uses rgb images
    results=hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:    # for each hand from multihands
            for id,lm in enumerate(handLms.landmark):   # to give id number to each landmark of the hand for each frame and get x,y,z coordinates whenever hand is detected.
                # print(id,lm)                            #this will print the values of landmark coordinates in decimal format

                h, w, c=img.shape                           #get the height , width and channel values for the images
                cx,cy=int(lm.x*w),int(lm.y*h)               #getting width and height of the image in integer format
                print(id,cx,cy)                             #print id,width and height in decimal
                if id==0:           #highlight wrist joint with circle
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                elif id==4 or id==8 or id==12 or id==16 or id==20:  # highlight finger tips with circle
                    cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)          #draw dots by detecting hands from handLms from img and draw connections
    cTime=time.time()
    fps=1/(cTime-pTime)     #fps
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)   #frame rate display on screen

    cv2.imshow("Image",img)
    cv2.waitKey(1)