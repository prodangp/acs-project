import cv2
import numpy as np
path = r'C:\Users\info\Desktop\Info Anul III\ACS\Proiect'




def read_reshape(pgm):
    data = cv2.imread(pgm)
    data = np.reshape(data, (10304, 3))
    data = data.T
    return data[0]


def training_matrix():
    A=[]
    for i in range(1, 41):
        for j in range(1, 9):
            pgm_path = path + r'\db\s' + str(i) + "\\" + str(j) + '.pgm'
            img = read_reshape(pgm_path).astype(np.float64)
            A.append(img)

    A = np.array(A)
    return A


def find_person(k):
    return int(k/8) + 1

def avg_img(A):
    avg_img = [0.0] * len(A[0])
    avg_img = np.array(avg_img).astype(np.float64)
    for img in A:
        for pixel in range(0, len(img)):
            avg_img[pixel] += img[pixel]
    for pixel in range(0, len(avg_img)):
        avg_img[pixel] = avg_img[pixel] / len(A)
    return avg_img

def center_data(A):
    avg_image = avg_img(A)
    for img_no in range(0, len(A)):
        A[img_no] = A[img_no] - avg_image
    return A, avg_image




