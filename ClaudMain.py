import numpy as np
import time, cv2, os, imutils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

def detect_nd_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)
            
            if endY - startY > 0 and endX - startX > 0:
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                
                faces.append(face)
                locs.append((startX, startY, endX, endY))
    
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
        
    return (locs, preds)

# Initialize models
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("MaskDetection.keras")

# Try different camera indices
camera_index = 0
max_attempts = 3

while camera_index < max_attempts:
    try:
        print(f"Trying camera index {camera_index}")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            camera_index += 1
            continue
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = imutils.resize(frame, width=400)
            
            try:
                (locs, preds) = detect_nd_predict_mask(frame, faceNet, maskNet)
                
                for (box, pred) in zip(locs, preds):
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred
                    
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
                cv2.imshow("Frame", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                    
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
                
        camera_index += 1
        
    except Exception as e:
        print(f"Error with camera {camera_index}: {str(e)}")
        camera_index += 1
        
print("No working camera found")
cv2.destroyAllWindows()