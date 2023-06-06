from scipy.spatial import distance as dist
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, Response
import cv2
from flask import Flask, render_template, Response, stream_with_context
import time
import queue

MIN_DISTANCE = 500  # Minimum distance for social distancing violation

# create a Flask app instance
app = Flask(__name__)

log_queue = queue.Queue()

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

            # initialize list of faces, corresponding locations, and list of distances
            faces = []
            locs = []
            centroids = []
            labels_mask = []
            labels_dist = []

            # iterate over the detected faces
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]

                    # preprocess the face ROI
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
                    centroids.append((startX + (endX - startX) / 2, startY + (endY - startY) / 2))
                    labels_dist.append(0)

            # only make a predictions if at least one face was detected
            if len(faces) > 0:
                faces = np.array(faces, dtype="float32")
                preds = model.predict(faces, batch_size=32)

                for pred in preds:
                    i = np.argmax(pred)
                    labels_mask.append(i)

            # convert the centroids to a NumPy array
            centroids = np.array(centroids)

            # compute pairwise distances between all centroids
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two centroid pairs is less than the configured number of pixels
                    if D[i, j] < MIN_DISTANCE:
                        labels_dist[i] = 1
                        labels_dist[j] = 1

            print(f"Distance Labels: {labels_dist}")  # Print out the labels to debug

            # loop over the detected face locations and their corresponding indexes
            for ((startX, startY, endX, endY), i) in zip(locs, range(len(locs))):
                # draw a bounding box around the detected face
                cv2.rectangle(frame, (startX, startY), (endX, endY), dist_label[labels_dist[i]], 2)

                label_text = mask_label[labels_mask[i]]
                print(f"Initial label_text: {label_text}")  # Print out the initial label to debug

                if labels_dist[i] == 1:
                    label_text += " (Social Distancing Violation)"
                    print(f"Updated label_text: {label_text}")  # Print out the updated label to debug

                # Display the label of the face
                cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            dist_label[labels_dist[i]], 2)

            # show the output frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            dist_violation = 1 in labels_dist
            if dist_violation:  # if there's a violation, add a log message
                log_message = "Social Distancing Violation Detected"
                log_queue.put(log_message)
            else:
                log_message = "No violation detected"
                log_queue.put(log_message)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs', methods=['GET'])
def logs():
    def event_stream():
        while True:
            while not log_queue.empty():
                yield 'data: {}\n\n'.format(log_queue.get())
            time.sleep(0.1)  # wait for log_queue to be populated

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route('/about')
def about():
    """page overview page."""
    return render_template('overview.html')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
