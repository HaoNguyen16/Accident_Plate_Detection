### importing required libraries
import matplotlib.pyplot as plt
import torch
import cv2
import time
# import pytesseract
import re
import numpy as np
import easyocr

##### DEFINING GLOBAL VARIABLE
EASY_OCR = easyocr.Reader(['en'])  ### initiating easyocr
OCR_TH = 0.3


### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])


    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

def accident_detect(results, frame, classes):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")
    detected = []
    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.85:  ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)  ## BBOx coordniates
            text_d = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1, y1, x2, y2]

            if text_d == 'accident':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  ## BBox
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)  ## for text label background
                cv2.putText(frame, f"{text_d}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cropped_img = frame[y1:y2, x1:x2]

                # cv2.imwrite(f"./output/accident_detected.jpg",frame)
                detected.append(cropped_img)

    return detected, frame


### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, classes):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    plate_frame = []
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    for i in range(n):
        row = cord[i]

        if row[4] >= 0.85:  ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)  ## BBOx coordniates
            text_d = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1, y1, x2, y2]

            plate_num = recognize_plate_easyocr(img=frame, coords=coords, reader=EASY_OCR, region_threshold=OCR_TH)

            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  ## BBox
            cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)  ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cropped_img = frame[y1:y2, x1:x2]
            # file = open('./output/PlateNumber.txt', 'w')
            # for j in range(len(plate_num)):
            # file.write(f'{plate_num[j]}\n')
            # file.close()
            plate_frame.append(cropped_img)
            # cv2.imwrite(f"./output/plate_detected{i}.jpg",cropped_img)

    return frame, plate_frame


#### ---------------------------- function to recognize license plate --------------------------------------


# function to recognize license plate numbers using Tesseract OCR
def recognize_plate_easyocr(img, coords, reader, region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]  ### cropping the number plate from the whole image

    ocr_result = reader.readtext(nplate)

    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold=region_threshold)

    if len(text) == 1:
        text = text[0].upper()
    return text


### to filter out wrong detections 

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]

    plate = []
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


### ---------------------------------------------- Main function -----------------------------------------------------

def main(vid_path=None, vid_out=None):
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model = torch.hub.load('./yolov5-master', 'custom', source='local', path='detection_best.pt',
                           force_reload=True)  ### The repo is stored locally
    model2 = torch.hub.load('./yolov5-master', 'custom', source='local', path='accident.pt',
                            force_reload=True)
    classes2 = model2.names
    classes = model.names  ### class names in string format


    if vid_path != None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)

        #if vid_out:  ### creating the video writer if video output path is given

            ##by default VideoCapture returns float instead of int
            ##width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ##height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ##fps = int(cap.get(cv2.CAP_PROP_FPS))
            ##codec = cv2.VideoWriter_fourcc(*'mp4v')  ##(*'XVID')
            ##out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
        frame_total = cap.get((cv2.CAP_PROP_FRAME_COUNT))
        frame_no = 1
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret and frame_no % 1 == 0:
                # print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model2)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                accident_frame, frame = accident_detect(results, frame, classes=classes2)


                for i in range(len(accident_frame)):
                    results2 = detectx(accident_frame[i], model=model)  ### DETECTION HAPPENING HERE
                    accident_frame[i], plate_frame = plot_boxes(results2, accident_frame[i], classes=classes)
                    # plate_detected.append(accident_frame[i])
                    # plate_crop.append(plate_frame[i])
                    accident_frame[i] = cv2.resize(accident_frame[i], [450, 450], interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(f'accident{i}', accident_frame[i])
                    # cv2.imwrite(f"./output/accident_frame{i}.jpg", accident_frame[i])
                    if len(plate_frame):
                        for j in range(len(plate_frame)):
                            cv2.imshow(f'plate detected{j}', plate_frame[j])
                            #cv2.imwrite(f'./output/plate{j}.jpg', plate_frame[j])
                cv2.imshow("normal_frame", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    # stop = 0
                    break
                if frame_no >= frame_total:
                    break
                frame_no += 1



        ## closing all windows'''
        cv2.destroyAllWindows()


### -------------------  calling the main function-------------------------------#


main(vid_path="./test_images/Test_accident1.mp4")  ### for custom video
# main(vid_path=0, vid_out="webcam_facemask_result.mp4")  #### for webcam

