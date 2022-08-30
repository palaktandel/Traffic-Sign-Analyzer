import numpy as np
import cv2
import time
import os
from tensorflow.keras.models import load_model



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def main():

    output_folder_path = "images/output_images"
    HEIGHT = 32
    WIDTH = 32

    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")

    classes = []
    with open("signs.names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    #get last layers names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    check_time = True
    confidence_threshold = 0.7
    font = cv2.FONT_HERSHEY_SIMPLEX



    detection_confidence = 0.5
    print("Type '1' to start video capture or '0' to anaylize the pre-recorded video")
    choice = input()
    if choice == '1':
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture("video..mp4")

    font = cv2.FONT_HERSHEY_SIMPLEX

    classification_model = load_model('traffic.h5') #load mask detection model
    classes_classification = []
    with open("signs_classes.txt", "r") as f:
        classes_classification = [line.strip() for line in f.readlines()]
        i = 0
        frame_skip = 1
        frame_count = 0
        while(True):
            i = i + 1
            ret, img = cap.read()
            if ret == False:
                continue
            frame_count = frame_count + 1
            if frame_count % frame_skip != 0:
                continue
            frame_count = 0
            height, width, channels = img.shape



            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                    crop_img = img[y:y+h, x:x+w]
                    if len(crop_img) >0:
                        try:
                            crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                        except:
                            continue
                        crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                        prediction = np.argmax(classification_model.predict(crop_img))
                        # print(prediction)
                        label = str(classes_classification[prediction])
                        print("Detected sign ---> " , label)
                        # os.system("play -nq -t alsa synth 1 sine 440")
                        img = cv2.putText(img, label, (x, y), font, 0.5, (255,0,0), 2)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord ('q'):
               break
        cap.release()
        cv2.destroyAllWindows()

main()
