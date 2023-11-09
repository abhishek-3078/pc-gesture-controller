import cv2
import pyautogui
import mediapipe as mp
import numpy as np
import threading
import time
import subprocess


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
screenWidth, screenHeight = pyautogui.size()

left_click_pressed = False

scrolling = False

value = 0
t1,t2 = 0,0

prev_x, prev_y = None, None

def get_distance(x,y):
    return np.linalg.norm(np.array(x) - np.array(y))

def condition_satisfied(dist_left_click):
    if dist_left_click < 0.05:
        return True  
    else :
        return False

def check_condition_wrapper(dist_left_click):
    return lambda: check_condition(dist_left_click)

def check_condition(dist_left_click):
    global value
    if condition_satisfied(dist_left_click):  
        value += 1
    else:
        value = 0
        return

    duration = 1  
    threading.Timer(duration, check_condition_wrapper(dist_left_click)).start()

def moveMouse(wrist_x,wrist_y):
    global prev_x,prev_y

    mousePositionX = int((screenWidth) * wrist_x)
    mousePositionY = int(screenHeight * wrist_y)

    point1 = (thumb_x, thumb_y)
    point2 = (index_x, index_y)
    dist_left_click = np.linalg.norm(np.array(point1) - np.array(point2))

    if dist_left_click < 0.05:

        if prev_x is not None and prev_y is not None:
            diff_x, diff_y = mousePositionX - prev_x, mousePositionY - prev_y
            x,y = pyautogui.position()
            if(x+diff_x*2>=screenWidth or x+diff_x*2<0 or y+diff_y<0 or y+diff_y>=screenHeight):
                #to preven Corener Exception
                return
            pyautogui.moveRel(diff_x*2, diff_y*2, duration=0.1)
            
    prev_x, prev_y = mousePositionX, mousePositionY

    


def left_click(distance):
    global value
    global t1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'LEFT CLICK',(10,100), font, 4,(255,255,255),2,cv2.LINE_AA)
    pyautogui.leftClick()
    # Time2
    # print("LEFT CLICK")
    t1 = time.time()
    check_condition(distance)
    if value > 6:
        pyautogui.doubleClick()
        value = 0

    cv2.waitKey(1)

def right_click():
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'RIGHT CLICK',(10,100), font, 4,(255,255,255),2,cv2.LINE_AA)
    
    pyautogui.rightClick()
    print("RIGHT CLICK")

    cv2.waitKey(1)

def check_lock(thumb,ring):
    #peace sign from left hand .
    dist=get_distance((thumb.x,thumb.y),(ring.x,ring.y))
    if(dist<0.05):
        return 1



while True:
    ret, frame = cap.read()
    frameHeight, frameWidth, _ = frame.shape
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
       
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
           
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            middle_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            middle_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
            wrist_x = hand_landmarks.landmark[9].x
            wrist_y = hand_landmarks.landmark[9].y


            # print(hand_landmarks.landmark)

            if hand_landmarks.landmark[0].x < hand_landmarks.landmark[5].x:
                hand_id = 0  # Left hand
            else:
                hand_id = 1  # Right hand

                        

            # Performing tasks based on the hand ID
            if hand_id == 1:
                
                if thumb_y < middle_y:
                    hand_right_gesture = 'pointing up'
                    if not scrolling:
                        scrolling = True
                        scroll_start = pyautogui.position()
                elif thumb_y > middle_y:
                    hand_right_gesture = 'pointing down'
                    if scrolling:
                        scrolling = False
                        scroll_start = None
                        pyautogui.scroll(-700)
                    else:
                        hand_right_gesture = "other"

                if scrolling:
                    current_pos = pyautogui.position()
                    # pyautogui.moveTo(current_pos[0], current_pos[1] + (pyautogui.position()[1] - scroll_start[1]))

                moveMouse(wrist_x,wrist_y)


            elif hand_id == 0:
                # Task for the left hand
                should_lock_screen=check_lock(hand_landmarks.landmark[4],hand_landmarks.landmark[15]);
                if(should_lock_screen):
                    print("locking the screen")
                    subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])

                point1 = (thumb_x, thumb_y)
                point2 = (index_x, index_y)
                point3 = (middle_x,middle_y)

                dist_left_click = np.linalg.norm(np.array(point1) - np.array(point2))

                dist_right_click = np.linalg.norm(np.array(point1) - np.array(point3))

                if thumb_y < middle_y:
                    hand_gesture = 'pointing up'
                    if not scrolling:
                        scrolling = True
                        scroll_start = pyautogui.position()
                elif thumb_y > middle_y:
                    hand_gesture = 'pointing down'
                    if scrolling:
                        scrolling = False
                        scroll_start = None
                        pyautogui.scroll(700)
                    else:
                        hand_gesture = "other"

                if scrolling:
                    current_pos = pyautogui.position()
                    # pyautogui.moveTo(current_pos[0], current_pos[1] + (pyautogui.position()[1] - scroll_start[1]))

                if dist_left_click < 0.05:  
                    # Time1
                    t2 = time.time()
                    left_click(dist_left_click)
                    print("Delay : ",t1-t2," ms")

                if dist_right_click < 0.05:  
                    right_click()
                
            
        cv2.imshow("Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()