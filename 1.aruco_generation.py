import numpy as np
import cv2
import os
VIDEO_EXT = ['.jpg']

def get_video_list(path):
    video_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in VIDEO_EXT:
                video_names.append(apath)
    return video_names

def detector_marker(img, aruco_dict, parameters, cameraMatrix, distCoeffs):
    fname = img.split(os.sep)[-1]
    frame = cv2.imread(img)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # lists of ids and the corners beloning to each id# We call the function 'cv2.aruco.detectMarkers()'
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

    # Draw detected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))

    # Draw rejected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))

    if ids is not None:
        # rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in corners.
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 1)

    cv2.imwrite(f'./result/{fname}')

def generator(aruco_dict):
    for id in range(100):
        img_size = 700 # 定义最终图像的大小
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, img_size)
        cv2.imwrite(f'./raw_4x4_250/{str(id).zfill(6)}.jpg',marker_img)



if __name__ =='__main__':

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    # aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

    # Create parameters to be used when detecting markers:
    generator(aruco_dict)