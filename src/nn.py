from tools import *
from statistics import mode


def nn(test_img, A, norm):
    z = []
    for img in A:
        if norm == 'cos':
            cos = np.dot(test_img, img) / (np.linalg.norm(test_img) * np.linalg.norm(img))
            z.append(1 - cos)
        else:
            z.append(np.linalg.norm(img - test_img, norm))
    x = np.argmin(z)
    return find_person(x)


def knn(test_img, A, norm, k=5):
    z = []
    for img in A:
        if norm == 'cos':
            cos = np.dot(test_img, img) / (np.linalg.norm(test_img) * np.linalg.norm(img))
            z.append(1-cos)
        else:
            z.append(np.linalg.norm(img - test_img, norm))
    indices_sort = np.argsort(z)
    persons = []
    for i in range(0, k):
        persons.append(find_person(indices_sort[i]))
    return mode(persons)

