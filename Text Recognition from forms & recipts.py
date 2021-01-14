import cv2
import numpy as np
import pytesseract
import os
# C:\Program Files\Tesseract-OCR
per = 25
pixelThreshold=100
roi = [[(173, 70), (374, 104), 'text', 'Adhaar Number'], [(104, 131), (365, 162), 'text', 'Name'],
 [(117, 188), (391, 222), 'text', 'District'], [(104, 243), (396, 280), 'text', 'State'],
  [(55, 341), (76, 353), 'box', 'Male'], [(170, 337), (193, 353), 'box', 'Female'],
   [(54, 397), (77, 412), 'box', 'Age 0-18'],
   [(171, 398), (194, 414), 'box', 'Age 19-35'], [(282, 396), (308, 414), 'box', 'Age 36-50'],
  [(384, 397), (407, 413), 'box', 'Age 51-60'], [(470, 399), (494, 415), 'box', 'Age 60 Above'],
 [(54, 453), (78, 467), 'box', 'Obese Yes'], [(192, 451), (216, 467), 'box', 'Obese No'],
  [(56, 507), (80, 523), 'box', 'Heart Disease Yes'], [(192, 508), (216, 522), 'box', 'Heart Disease No'],
   [(57, 567), (80, 584), 'box', 'Lung Problem Yes'], [(196, 569), (218, 582), 'box', 'Lung Problem No'],
    [(55, 625), (79, 642), 'box', 'Muscle/ Joint pain'], [(210, 628), (232, 641), 'box', 'Tiredness'],
     [(321, 628), (345, 644), 'box', 'Headache'], [(427, 625), (451, 643), 'box', 'None'],
      [(54, 654), (78, 669), 'box', 'Fever'], [(147, 652), (171, 668), 'box', 'Mild Allergies'],
     [(54, 713), (77, 727), 'box', 'Covid Positive'], [(191, 713), (214, 728), 'box', 'Covid Negative']]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
imgQ = cv2.imread('Query.png')
h,w,c = imgQ.shape
imgQ = cv2.resize(imgQ,(w//1,h//1))
orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)
path = 'C:\\Users\\nikhi\\OneDrive\\Documents\\Custom Office Templates\\PycharmProjects\\Text Recognition OCR\\UserForms'
myPicList = os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img = cv2.imread(path +"/"+y)
    img = cv2.resize(img, (w // 1, h // 1))
    #cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:20],None,flags=2)
    #cv2.imshow(y, img)
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,2.0)
    imgScan = cv2.warpPerspective(img,M,(w,h))
    cv2.imshow(y, imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []

    print(f'################## Extracting Data from Form {j}  ##################')

    for x,r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':
            print('{} :{}'.format(r[3],pytesseract.image_to_string(imgCrop)))
            myData.append(pytesseract.image_to_string(imgCrop))
        if r[2] =='box':
            imgGray = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray,17,255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
            if totalPixels>pixelThreshold: totalPixels =1;
            else: totalPixels=0
            print(f'{r[3]} :{totalPixels}')
            myData.append(totalPixels)
        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
                    cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)
    with open('DataOutput1.csv','a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write('\n')

    #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    print(myData)



    cv2.imshow(y,imgShow)



#cv2.imshow("KeyPoints", impKp1)
cv2.imshow("Output",imgQ)
cv2.waitKey(0)
