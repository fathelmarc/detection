import cv2
import numpy as np
class Detect:
    classesFile = '/home/marcel/darknet/cfg/coco.names'
    classNames = []
    with open(classesFile, 'rt')as f:
        classNames = f.read().rstrip('\n').split('\n')
    modelConfiguration = '/home/marcel/darknet/cfg/yolov4.cfg'
    modelWeights = '/home/marcel/darknet/yolov4.weights'
    net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def __init__(self) -> None:
        pass
    def findobject(outputs, img):
        hT,wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        confThreshold = 0.3
        nmsThreshold = 0.3
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidance = scores[classId]
                if confidance > confThreshold:
                    w,h = int(det[2]*wT), int(det[3]*hT)
                    x,y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                    bbox.append([x,y,w,h])
                    classIds.append (classId)
                    confs.append(float(confidance))
        print(len(bbox))
        indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
        print(indices)
        for i in indices:
            i = i
            if len(bbox)>1:
                box = bbox[1]
                x,y,w,h = box[0],box[1],box[2],box[3]
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,f'{Detect.classNames[classIds[1]].upper()}{int(confs[i]*100)}%',
                                (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)