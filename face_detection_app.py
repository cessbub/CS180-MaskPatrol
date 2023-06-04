# # import necessary libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# from flask import Flask,render_template,Response
# import cv2
import numpy as np

# import necessary libraries
from flask import Flask,render_template,Response
import cv2

# create a Flask app instance
app=Flask(__name__)

# set up the camera as the video source
camera=cv2.VideoCapture(0)

# define a function to generate video frames
def gen_frames():

    # load the prediction model
    mask_label = {0:'MASK', 1:'UNCOVERED CHIN', 2:'UNCOVERED NOSE', 3:'UNCOVERED NOSE AND MOUTH', 4:"NO MASK"}
    dist_label = {0:(0,255,0),1:(255,0,0)}

    model = load_model("mask_detection.h5")
    while True:
        
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        
        else:
            detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            # eye_cascade =cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            # Draw The rectangle arounf the each face
            label = [0 for i in range(len(faces))]

            #draws rectangle on the faces
            for (x, y, w, h ) in faces:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2) 

            for i in range(len(faces)):
                (x,y,w,h) = faces[i]
                crop = frame[y:y+h,x:x+w]
                crop = cv2.resize(crop,(224,224))
                crop = img_to_array(crop)
                crop = preprocess_input(crop)
                crop = np.array(crop, dtype="float32")

                # print(crop)
                # print(crop.shape)

                mask_result = model.predict(np.expand_dims(crop, axis=0), batch_size=1)

                print(mask_result.argmax())
                cv2.putText(frame,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),dist_label[label[i]],1)    
                    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
# define the root URL endpoint
@app.route('/')
def index():
    
    # render the index.html template
    return render_template('index.html')

# define a video feed URL endpoint
@app.route('/video_feed')
def video_feed():
    
    # return the video stream as a Flask response
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
 
# Start the Flask app with debugging enabled    
if __name__=='__main__':
    app.run(debug=True)        
                    
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# from mtcnn import MTCNN

# app = Flask(__name__)

# camera = cv2.VideoCapture(0)

# def gen_frames():
#     mask_label = {0: 'MASK', 1: 'UNCOVERED CHIN', 2: 'UNCOVERED NOSE', 3: 'UNCOVERED NOSE AND MOUTH', 4: 'NO MASK'}
#     dist_label = {0: (0, 255, 0), 1: (255, 0, 0)}

#     model = load_model('mask_detection.h5')
#     detector = MTCNN()

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = detector.detect_faces(frame)
#             # print(faces)

#             for face in faces:
#                 print(face['box'])
#                 x, y, w, h = face['box']
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#             #     crop = frame[y:y + h, x:x + w]
#             #     crop = cv2.resize(crop, (224, 224))
#             #     crop = img_to_array(crop)
#             #     crop = preprocess_input(crop)
#             #     crop = np.expand_dims(crop, axis=0)

#             #     mask_result = model.predict(crop, batch_size=1)
#             #     label = mask_result.argmax()

#             #     cv2.putText(frame, mask_label[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#             #                 dist_label[label], 2)
#             #     cv2.rectangle(frame, (x, y), (x + w, y + h), dist_label[label], 1)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)                    
                                
            
            