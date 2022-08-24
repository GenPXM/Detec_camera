import numpy as np
import cv2
import os
import time
import base64
import json
import sys
import smtplib

save_dir = "./img/"

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
max_image_number = 5

TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[3]





def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)

    return kernel


def getFilter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, getKernel("dilation"), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel("closing"), iterations=2)
        dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)


        return dilation


def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Detector inválido")
    sys.exit(1)


def main (args):
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture("input camera")
    minArea = 400
    bg_subtractor = getBGSubtractor(BGS_TYPE)

    while(True):
        ret, frame = cap.read()
        ret1, frame1 = cap2.read()

        frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
        frame1 = cv2.resize(frame1, (0, 1), fx=0.50, fy=0.50)
        #cv2.imshow('Capturing0', frame)
        #cv2.imshow('Capturing1', frame1)
        bg_mask = bg_subtractor.apply(frame)
        bg_mask1 = bg_subtractor.apply(frame1)

        bg_mask = getFilter(bg_mask, "opening")
        bg_mask1 = getFilter(bg_mask1, "opening")

        bg_mask = cv2.medianBlur(bg_mask, 5)
        bg_mask1 = cv2.medianBlur(bg_mask1, 5)


        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (contours1, hierarchy1) = cv2.findContours(bg_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= minArea:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (10, 30), (250, 55), (255, 0, 0), -1)
                cv2.putText(frame, "Movimento Detectado", (10, 50), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
                cv2.drawContours(frame, cnt, -1, TRACKER_COLOR, 3)
                cv2.drawContours(frame, cnt, -1, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), TRACKER_COLOR, 3)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
                img_file_path = save_dir + str(time.time()) + ".jpg"
                cv2.imwrite(img_file_path, frame)
                img_list = os.listdir(save_dir)
                img_list.sort()
                if len(img_list) > max_image_number:
                    # print("before", img_list)
                    for i in range(0, len(img_list)):
                        os.remove(save_dir + img_list[i])
                        # print('path', save_dir + img_list[i])
                        img_list.pop(0)
                        # print("before222", img_list)
                        if len(img_list) == max_image_number:
                            break
            imgs_data = {}
            # print("bbb", img_list)
            img_list.sort(reverse=True)
            for v in img_list:
                with open(save_dir + v, mode='rb') as f:
                    imgs_data[v] = base64.b64encode(f.read()).decode("utf-8")
            # print(imgs_data)

            with open("imgs.json", "w") as f:
                json.dump(imgs_data, f)
            # time.sleep(1)



        for cnt in contours1:
            area1 = cv2.contourArea(cnt)
            if area1 >= minArea:
                cv2.rectangle(frame1, (10, 30), (250, 55), (255, 0, 0), -1)
                cv2.putText(frame1, "Movimento Detectado", (10, 50), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
                cv2.drawContours(frame1, cnt, -1, TRACKER_COLOR, 3)
                cv2.drawContours(frame1, cnt, -1, (255, 255, 255), 1)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), TRACKER_COLOR, 3)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 255, 255), 1)
                img_file_path = save_dir + str(time.time()) + ".jpg"
                cv2.imwrite(img_file_path, frame1)
                img_list = os.listdir(save_dir)
                img_list.sort()
                if len(img_list) > max_image_number:
                    # print("before", img_list)
                    for i in range(0, len(img_list)):
                        os.remove(save_dir + img_list[i])
                        # print('path', save_dir + img_list[i])
                        img_list.pop(0)
                        # print("before222", img_list)
                        if len(img_list) == max_image_number:
                            break
            imgs_data = {}
            # print("bbb", img_list)
            img_list.sort(reverse=True)
            for v in img_list:
                with open(save_dir + v, mode='rb') as f:
                    imgs_data[v] = base64.b64encode(f.read()).decode("utf-8")
            # print(imgs_data)

            with open("imgs.json", "w") as f:
                json.dump(imgs_data, f)
            # time.sleep(1)


        result = cv2.bitwise_and(frame, frame, mask=bg_mask)
        result1 = cv2.bitwise_and(frame1, frame1, mask=bg_mask1)


        cv2.imshow('Frame', frame)
        cv2.imshow('Frame1', frame1)
        cv2.imshow('Mask', result)
        cv2.imshow('Mask2', result1)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
