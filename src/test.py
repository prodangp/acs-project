
from nn import *
from eigenfaces import *
import lanczos
import tensor_a1

test_img_per_person = 2
persons_no = 40
img_no = test_img_per_person * persons_no

algorithms = {'Nearest neighbours': nn, 'k-Nearest neighbours': knn, 'Eigenfaces': eigenfaces,
              'Eigenfaces 2': eigenfaces_reduced, 'Lanczos': lanczos.test, 'Tensori A1': tensor_a1.tensor_a1}
norms = {'Euclidean': 2, 'Manhattan': 1, 'Infinity': np.inf, 'Cosine': 'cos'}
# A = training_matrix()
# algorithm = eigenfaces_reduced
# rr = 0
# total_time = 0
#
# for norm in [2, 1, np.inf, 'cos']:
#     rr = 0
#     total_time = 0
#     for p in range(1, persons_no + 1):
#         test_images = [path + r'\db\s' + str(p) + r'\9.pgm', path + r'\db\s' + str(p) + r'\10.pgm']
#         for test_image_path in test_images:
#             test_image = read_reshape(test_image_path).astype(np.float64)
#             tic = time()
#             result = algorithm(test_image, A, norm, k=140)
#             toc = time()
#             total_time += toc - tic
#             if result == p:
#                 rr += 1/img_no
#     print(round(rr*100, 2), '%')
#     if norm == 2:
#         from tensor_a1 import ppt
#         test_time = total_time - ppt
#     else:
#         test_time = total_time
#     print(test_time/img_no * 1000, 'ms')
#     test_time = 0


def test_image(image, algorithm, norm, ti):
    image = read_reshape(image).astype(np.float64)
    if algorithm == 'Tensori A1':
        A = training_tensor(TI=ti)
    else:
        A = training_matrix(TI=ti)
    p = algorithms[algorithm](image, A, norms[norm])
    if p<0:
        return path + r'\db\unknown.png'
    return path + r'\db\s' + str(p) + r'\1.pgm'
