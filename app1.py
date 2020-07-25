import cv2
import numpy as np
import time
import math
from flask import Flask,render_template,Response, session
from markupsafe import Markup
from flask_socketio import SocketIO, emit


app = Flask(__name__,static_url_path = "/",static_folder = "template",template_folder='template')
app.config["TEMPLATES_AUTO_RELOAD"] = True
socketio = SocketIO(app)
msg=""

@app.route('/')
def index():    
    return render_template('./CarGame.html')

def gen(): 
    global msg
    cap = cv2.VideoCapture(0)
         
    while(cap.isOpened()):
            
            ret, frame = cap.read()
            #frame=cv2.flip(frame,1)
            kernel = np.ones((3,3),np.uint8)
            
            #define region of interest
            roi=frame[0:350, 0:350]
            
            
            cv2.rectangle(frame,(50,50),(350,350),(0,255,0),0)    
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            
             
        # define range of skin color in HSV
            lower_skin = np.array([0,20,70], dtype=np.uint8)
            upper_skin = np.array([20,255,255], dtype=np.uint8)
            
         #extract skin colur imagw  
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
       
            
        #extrapolate the hand to fill dark spots within
            mask = cv2.dilate(mask,kernel,iterations = 4)
            
        #blur the image
            mask = cv2.GaussianBlur(mask,(5,5),100) 
            
            
            
        #find contours
            contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
       #find contour of max area(hand)
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            
        #approx the contour a little
            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,epsilon,True)
           
            
        #make convex hull around hand
            hull = cv2.convexHull(cnt)
            
         #define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
          
        #find the percentage of area not covered by hand in convex hull
            arearatio=((areahull-areacnt)/areacnt)*100
        
         #find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
            
        # l = no. of defects
            l=0
            
        #code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt= (100,180)
                
                
                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a+b+c)/2
                ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                
                #distance between point and convex hull
                d=(2*ar)/a
                
                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                
            
                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d>10:
                    l += 1
                    cv2.circle(roi, far, 3, [255,0,0], -1)
                
                #draw lines around hand
                cv2.line(roi,start, end, [0,255,0], 2)
                
                
            l+=1
            
            #print corresponding gestures which are in their ranges
            font = cv2.FONT_HERSHEY_SIMPLEX
            if l==1:
                if areacnt<2000:
                    cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    msg=""
                else:
                    if arearatio<12:
                        cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        msg="ArrowDown"
                    #elif arearatio<17.5:
                    #    cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        msg="ArrowLeft"
                        
            elif l==2:
                cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                msg="ArrowRight"
            elif l==3:
                  if arearatio<27:
                        cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        msg="ArrowUp"
                  else:
                        cv2.putText(frame,'Do it proper',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        msg=""
            elif l==4 or l==5 or l==6:
                cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                msg="ArrowUp"
                        
            else :
                cv2.putText(frame,'reposition it',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                msg=""
                
           
            fff = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fff + b'\r\n')
            
            
            k = cv2.waitKey(20)
            if k == 27:
                cv2.destroyAllWindows()
                cap.release()   
                print('webcam released !')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('greet')
def handle_my_custom_event():
    print("Emited from client")
    global msg
    emit('myResponse', msg)


if __name__ == '__main__':
	socketio.run(app)
    #app.run(debug=False)
