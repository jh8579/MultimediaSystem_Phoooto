import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import time

def extractImages(pathIn, imgs_list,frames):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    print(frames)
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * (frames)))  # 1000->1frame 100->10frame  100/6->60frame
        success, image = vidcap.read()

        if (success):
            imgs_list.append(image)
        count = count + 1
    print(count,"프레임 절단")


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def blurry_detect(imgs_list):
    count = []
    temp = 0

    for image in imgs_list:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            print(e)
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"
        if fm < 50:
            count.append(temp)
            text = "Blurry"
        temp = temp + 1

    for i in reversed(count):
        imgs_list.pop(i)


def make_panorama(imgs_list,pathout,frames):
    stitcher = cv2.createStitcher(False)
    blurry_detect(imgs_list)

    # 첫번째 초기 이미지 pop
    temp_img = imgs_list.pop(0)

    # result 초기화
    result = stitcher.stitch([temp_img,temp_img])

    # count 초기화
    cnt = 1
    final=result
    # 모든 img 한번씩 stitch
    for img in imgs_list:

        # count 출력 및 증가
        print(cnt)
        cnt = cnt + 1

        # result[0] 코드 번호
        # result[1] img value

        # result value 값이 있으면 result value 값과 현재 img stitch
        # result value 값이 없으면 초기 이미지와 현재 img stitch

        if (np.any(result[1] != None)):
            result = stitcher.stitch([img, result[1]])
        else:
            result = stitcher.stitch([temp_img, img])

        # 에러 코드 출력
        if (result[0] != 0):
            print("error for code", result[0])

        else:

            result = draw_line(result[1])
            final = result
            print("done")

    cv2.imshow(str(frames)+" Frames Result", final[1])
    '''error code 
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    '''

    now = time.localtime()
    file_name = pathout + "\\"+str(frames)+"FPS_%04d-%02d-%02d_%02d-%02d-%02d.jpg" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    cv2.imwrite(file_name,final[1])  # save frame as JPEG file


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mkdir(pathOut):
    try:
        if not (os.path.exists(pathOut)):
            os.makedirs(os.path.join(pathOut))
    except OSError as e:
        print(e)

# 사진 늘리기
def find_line():
    img = cv2.imread("./data/result/result.jpg")
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.Canny(imgray, 100, 200, 3)

    ret, thresh = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY_INV)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgray, contours[0], 1, (0, 255, 0))

    cv2.imshow('result2.jpg', imgray)
    cv2.imshow('result2.jpg', im2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def length_point(x, y):
    return abs(x[0] - y[0]) * abs(x[0] - y[0]) + abs(x[1] - y[1]) * abs(x[1] - y[1])


def draw_line(img1):
    img = img1.copy()
    img2 = img1.copy()
    img3 = img1.copy()

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY)
    edge, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areaArray = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)
    sorted_data = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    cnt = sorted_data[0][1]

    epsilon = 0.000001 * cv2.arcLength(cnt, False)
    approx = cv2.approxPolyDP(cnt, epsilon, False)

    cv2.drawContours(img2, [approx], 0, (0, 255, 0), 3)

    cv2.waitKey(0)

    height, width, channels = img2.shape
    maxTopLeft = [width, 0]
    maxbottomLeft = [0, height]

    min = 1000000
    max = -100

    topLeft = [0, 0]
    topRight = [0, 0]
    bottomLeft = [0, 0]
    bottomRight = [0, 0]

    for j in approx:
        i = j[0]
        if length_point(i, [0, 0]) < min:
            min = length_point(i, [0, 0])
            topLeft = i

    min = 1000000

    for j in approx:
        i = j[0]
        if length_point(i, maxTopLeft) < min:
            min = length_point(i, maxTopLeft)
            topRight = i

    min = 1000000

    for j in approx:
        i = j[0]
        if length_point(i, maxbottomLeft) < min:
            min = length_point(i, maxbottomLeft)
            bottomLeft = i

    for j in approx:
        i = j[0]
        if length_point(i, [0, 0]) > max:
            max = length_point(i, [0, 0])
            bottomRight = i

    return [0, warp_affine(topLeft, topRight, bottomLeft, bottomRight, img1)]


def warp_affine(topLeft, topRight, bottomLeft, bottomRight, img):
    # 네 개의 꼭지점 파라미터로 받음

    # 변환되기 전 좌표
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 네 개의 좌표를 이용하여 두개의 너비 두개의 높이를 구한다
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    minWidth = min([w1, w2])
    minHeight = min([h1, h2])

    # 변환될 대상 좌표
    pts2 = np.float32([[0, 0], [minWidth - 1, 0], [minWidth - 1, minHeight - 1], [0, minHeight - 1]])


    M = cv2.getPerspectiveTransform(pts1, pts2)
    result1 = cv2.warpPerspective(img, M, (int(minWidth), int(minHeight)))
    return result1


def run():
    print("start")
    arg = ["default", "100"]
    root = tk.Tk()
    root.title("Photto")
    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    def path():

        tk.Tk().withdraw()  # Close the root window
        pathin = filedialog.askopenfilename()
        print(pathin)
        arg[0] = pathin
    filemenu.add_command(label="Open", command=path)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=filemenu)
    lbl = tk.Label(root, text="Input FPS")
    lbl.grid(row=0, column=0)
    txt = tk.Entry(root)
    txt.grid(row=0, column=1)
    lbl_fin = tk.Label(root, text="")
    lbl_fin.grid(row=1, column=1)

    def clicked():
        if(arg[0]!="default"):
            res = txt.get() + " FPS"
            arg[1] = txt.get()
            lbl_fin.configure(text=res)
            imgs_list = []
            pathin = arg[0]
            print(arg[0])
            pathout=os.path.basename(pathin)
            pathout = "./result/"+pathout[:-4]
            print(pathout)
            frames = int(arg[1])
            #frames=1000//frames
            mkdir(pathout)
            extractImages(pathin,  imgs_list, frames)
            make_panorama(imgs_list,pathout,frames)
        else :
            messagebox.showerror("Error","Open File")

    processbt = tk.Button(root, text="Make_Panorama", width=15, command=clicked)
    processbt.grid(row=0, column=2)
    root.config(menu=menubar)
    root.mainloop()


# main 함수
if __name__ == "__main__":
    run()