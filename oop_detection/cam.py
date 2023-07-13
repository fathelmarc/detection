from Detectlib import *
        
cap = cv2.VideoCapture(0)
w = 640
h = 480
# den = Detect()

while True:
    done , img = cap.read()    

    blob = cv2.dnn.blobFromImage(img,1/255,(w,h),[0,0,0],1,crop=False)
    Detect.net.setInput(blob)

    layerNames = Detect.net.getLayerNames()

    outputNames = [layerNames[i-1] for i in Detect.net.getUnconnectedOutLayers()]

    outputs = Detect.net.forward(outputNames)

    Detect.findobject(outputs,img)

    cv2.imshow('cam',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows