import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Image folder declared
path = 'images'
images = []
className = []
myList = os.listdir(path)
# append images in folder to array
for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    className.append(os.path.splitext(cl)[0])


# encode images to be compared
def encodeImage(faces):
    toEncode = []
    for img in faces:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        toEncode.append(encode)
    return toEncode


# variable to call the face encoder for later use
encodedFace = encodeImage(images)


# make function to read and write to file
def attendenceMarker(studentName):
    with open('Attendence.csv', 'r+') as f:  # r+ is used to read and write to file
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')  # splits based on comma
            nameList.append(entry[0])
        if studentName not in nameList:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{studentName}, {time}')  # write names into file


# open camera and encode webcam picture
capture = cv2.VideoCapture(0)
while True:
    _, webImage = capture.read()
    resizeImg = cv2.resize(webImage, (0, 0), None, 0.25, 0.25)  # resize image to 1/4 of image size
    resizeImg = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2RGB)  # converts image to rgb
    currentFrame = face_recognition.face_locations(resizeImg)  # getting current face location
    encodeCurFrame = face_recognition.face_encodings(resizeImg, currentFrame)

    for face, faceLocation in zip(encodeCurFrame, currentFrame):
        facesMatch = face_recognition.compare_faces(encodedFace, face)  # compare face images
        faceDistance = face_recognition.face_distance(encodedFace, face)  # face distance to draw rectangle around
        matchIndex = np.argmin(faceDistance)  # if multiple faces, face recognition occurs on closest
        if facesMatch[matchIndex]:
            name = className[matchIndex]  # get name of person from image name
            y1, x2, y2, x1 = faceLocation  # gets face location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # multiplies by 4 due to reducing size of image by 4
            cv2.rectangle(webImage, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # writes name on the webcam
            cv2.putText(webImage, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 0)
            attendenceMarker(name)
    cv2.imshow('Face Detector', webImage)
    c = cv2.waitKey(1)
    if c == 27:
        break
capture.release()
cv2.destroyAllWindows()


