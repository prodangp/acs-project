import time
from nn import *
from eigenfaces import *
import lanczos

test_img_per_person = 2
persons_no = 40
img_no = test_img_per_person * persons_no

stats = {'nn': {}}
A = training_matrix()
algorithm = lanczos.test
norm = 1
rr = 0
total_time = 0


for p in range(1, persons_no + 1):
    test_images = [path + r'\db\s' + str(p) + r'\9.pgm', path + r'\db\s' + str(p) + r'\10.pgm']
    for test_image_path in test_images:
        test_image = read_reshape(test_image_path).astype(np.float64)
        tic = time.time()
        result = algorithm(test_image, A, norm)
        toc = time.time()
        total_time += toc - tic
        if result == p:
            rr += 1/img_no
print(round(rr*100, 2), '%')
print(total_time/img_no * 1000, 'ms')



