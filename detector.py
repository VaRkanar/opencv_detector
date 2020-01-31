# you can detect: body,eye,car,faceandeye,face
import cv2

car_classifier=cv2.CascadeClassifier('haarcascade/cars.xml')
face_classifier=cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_classifier=cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')
body_classifier=cv2.CascadeClassifier('haarcascade/haarcascade_fullbody.xml')

def detector(img,detect,name=False):
    gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    face=face_classifier.detectMultiScale(gray,1.3,3)
    eye=eye_classifier.detectMultiScale(gray,1.1,3)
    body=body_classifier.detectMultiScale(gray,1.1,3)
    car=car_classifier.detectMultiScale(gray,1.1,3)
    
    if detect=='body':
        for (x,y,w,h) in body:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            if name==True:
                cv2.putText(img,detect,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    elif detect=='eye':
        for (x,y,w,h) in eye:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            if name==True:
                cv2.putText(img,detect,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    elif detect=='car':
        for (x,y,w,h) in car:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            if name==True:
                cv2.putText(img,detect,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    elif detect=='face':
        for (x,y,w,h) in face:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            if name==True:
                cv2.putText(img,detect,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    elif detect=='faceandeye':                       
        for (x,y,w,h) in face:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=img[y:y+h,x:x+w]
            eye=eye_classifier.detectMultiScale(roi_gray)
            if name==True:
                cv2.putText(img,detect,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            for (ex,ey,ew,eh) in eye:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,100,200),3)
                roi_color=cv2.flip(roi_color,1)
                roi_color=img
    else:
        print("No any classifier choose!")
    return img 

