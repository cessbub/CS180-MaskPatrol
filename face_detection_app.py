# # # import necessary libraries
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# # from flask import Flask,render_template,Response
# # import cv2
# import numpy as np

# # import necessary libraries
# from flask import Flask,render_template,Response
# import cv2

# # create a Flask app instance
# app=Flask(__name__)

# # set up the camera as the video source
# camera=cv2.VideoCapture(0)

# # define a function to generate video frames
# def gen_frames():

#     # load the prediction model
#     mask_label = {0:'MASK', 1:'UNCOVERED CHIN', 2:'UNCOVERED NOSE', 3:'UNCOVERED NOSE AND MOUTH', 4:"NO MASK"}
#     dist_label = {0:(0,255,0),1:(255,0,0)}

#     model = load_model("mask_detection.h5")
#     while True:
        
#         ## read the camera frame
#         success,frame=camera.read()
#         if not success:
#             break
        
#         else:
#             detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
#             # eye_cascade =cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
#             faces=detector.detectMultiScale(frame,1.1,7)
#             gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
#             # Draw The rectangle arounf the each face
#             label = [0 for i in range(len(faces))]

#             #draws rectangle on the faces
#             for (x, y, w, h ) in faces:
#                 cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2) 

#             for i in range(len(faces)):
#                 (x,y,w,h) = faces[i]
#                 crop = frame[y:y+h,x:x+w]
#                 crop = cv2.resize(crop,(224,224))
#                 crop = img_to_array(crop)
#                 crop = preprocess_input(crop)
#                 crop = np.array(crop, dtype="float32")

#                 # print(crop)
#                 # print(crop.shape)

#                 mask_result = model.predict(np.expand_dims(crop, axis=0), batch_size=1)

#                 print(mask_result.argmax())
#                 cv2.putText(frame,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
#                 cv2.rectangle(frame,(x,y),(x+w,y+h),dist_label[label[i]],1)    
                    
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
# # define the root URL endpoint
# @app.route('/')
# def index():
    
#     # render the index.html template
#     return render_template('index.html')

# # define a video feed URL endpoint
# @app.route('/video_feed')
# def video_feed():
    
#     # return the video stream as a Flask response
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
    
 
# # Start the Flask app with debugging enabled    
# if __name__=='__main__':
#     app.run(debug=True)        
                    
# import necessary libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, Response
import cv2

# create a Flask app instance
app = Flask(__name__)

# set up the camera as the video source
camera = cv2.VideoCapture(0)

# load the pre-trained face detection model
prototxt_path = 'face_detector_model/deploy.prototxt'
model_path = 'face_detector_model/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# load the mask detection model
mask_label = {0: 'MASK', 1: 'UNCOVERED CHIN', 2: 'UNCOVERED NOSE', 3: 'UNCOVERED NOSE AND MOUTH', 4: "NO MASK"}
dist_label = {0: (0, 255, 0), 1: (255, 0, 0), 2:(255, 0, 0), 3: (255, 0, 0), 4: (255, 0, 0)}
model = load_model("mask_detection.h5")

# define a function to generate video frames
def gen_frames():
    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # convert the frame to blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network to detect faces
            faceNet.setInput(blob)
            detections = faceNet.forward()
            print(detections.shape)

            # iterate over the detected faces
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]

                    # preprocess the face ROI
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    # print(face.shape)
                    # # perform mask detection on the face
                    mask_result = model.predict(np.expand_dims(face, axis=0), batch_size=1)
                    label_index = mask_result.argmax()

                    # # draw bounding box and label on the frame
                    label = mask_label[label_index]
                    color = (dist_label[label_index])
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # convert the frame to a JPEG image and yield it to the browser
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
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask app with debugging enabled
if __name__ == '__main__':
    app.run(debug=True)   
            
            