import cv2 as cv
import scipy as sc
import scipy.sparse
import numpy as np
import scipy.signal
import scipy.sparse.linalg

target = cv.imread("./imgs/source.jpg")
# source = cv.imread("./imgs/timg.jpg")[int(1440 / 2 - 500):int(1440 / 2 + 500),
#                                      int(2560 / 2 - 500):int(2560 / 2 + 500)]
source = cv.imread("./imgs/target.png")
print(source.shape)


# Get the left part of the equation Ax=b
def get_matA(rows, cols):
    # In my test case, rows and cols are supposed to be the same
    matA = sc.sparse.lil_matrix((rows * cols, rows * cols))

    matA.setdiag(-1, -1)
    matA.setdiag(-1, 1)
    matA.setdiag(-1, rows)
    matA.setdiag(-1, -rows)
    matA.setdiag(4, 0)
    matA = matA.tocsr()

    y = 0
    for x in range(1, 999):
        k = x + y * 1000
        matA.data[matA.indptr[k]:matA.indptr[k + 1]] = 0
        matA[k, k] = 1

    y = 999
    for x in range(1, 999):
        k = x + y * 1000
        matA.data[matA.indptr[k]:matA.indptr[k + 1]] = 0
        matA[k, k] = 1

    x = 0
    for y in range(1, 999):
        k = x + y * 1000
        matA.data[matA.indptr[k]:matA.indptr[k + 1]] = 0
        matA[k, k] = 1

    x = 999
    for y in range(1, 999):
        k = x + y * 1000
        matA.data[matA.indptr[k]:matA.indptr[k + 1]] = 0
        matA[k, k] = 1
    # y = 1
    # for x in range(2, 998):
    #     k = x + y * 1000
    #     matA[k, k] = 3
    #     matA[k, k - 1000] = 0
    # y = 998
    # for x in range(2, 998):
    #     k = x + y * 1000
    #     matA[k, k] = 3
    #     matA[k, k + 1000] = 0
    # x = 1
    # for y in range(2, 998):
    #     k = x + y * 1000
    #     matA[k, k] = 3
    #     matA[k, k - 1] = 0
    # x = 998
    # for y in range(2, 998):
    #     k = x + y * 1000
    #     matA[k, k] = 3
    #     matA[k, k - 1] = 0

    # y = 0
    # for x in range(1, 999):
    #     k = x + y * 1000
    #     matA[k, k] = 1
    #     matA[k, k + 1000] = 0
    #     matA[k, k + 1] = 0
    #     matA[k, k - 1] = 0
    # y = 999
    # for x in range(1, 999):
    #     k = x + y * 1000
    #     matA[k, k] = 1
    #     matA[k, k - 1000] = 0
    #     matA[k, k + 1] = 0
    #     matA[k, k - 1] = 0
    # x = 0
    # for y in range(1, 999):
    #     k = x + y * 1000
    #     matA[k, k] = 1
    #     matA[k, k - 1000] = 0
    #     matA[k, k + 1000] = 0
    #     matA[k, k + 1] = 0
    # x = 999
    # for y in range(1, 999):
    #     k = x + y * 1000
    #     matA[k, k] = 1
    #     matA[k, k - 1000] = 0
    #     matA[k, k + 1000] = 0
    #     matA[k, k - 1] = 0

    # y = 0
    # x = 0
    # k = x + y * 1000
    # matA[k, k] = 1
    # matA[k, k + 1000] = 0
    # matA[k, k + 1] = 0

    # y = 999
    # x = 0
    # k = x + y * 1000
    # matA[k, k] = 1
    # matA[k, k - 1000] = 0
    # matA[k, k + 1] = 0

    # y = 0
    # x = 999
    # k = x + y * 1000
    # matA[k, k] = 1
    # matA[k, k + 1000] = 0
    # matA[k, k - 1] = 0

    # y = 999
    # x = 999
    # k = x + y * 1000
    # matA[k, k] = 1
    # matA[k, k - 1000] = 0
    # matA[k, k - 1] = 0

    # y = 1
    # x = 1
    # k = x + y * 1000
    # matA[k, k] = 2
    # matA[k, k - 1000] = 0
    # matA[k, k - 1] = 0
    # y = 1
    # x = 998
    # k = x + y * 1000
    # matA[k, k] = 2
    # matA[k, k - 1000] = 0
    # matA[k, k + 1] = 0
    # y = 998
    # x = 1
    # k = x + y * 1000
    # matA[k, k] = 2
    # matA[k, k + 1000] = 0
    # matA[k, k - 1] = 0
    # y = 998
    # x = 998
    # k = x + y * 1000
    # matA[k, k] = 2
    # matA[k, k + 1000] = 0
    # matA[k, k + 1] = 0

    # print(matA.shape)

    return matA


def edit(source, targeting):
    tar = targeting
    target = targeting[1000:2000, 1000:2000]
    # source = source.flatten()
    # target = target.flatten()
    matA = get_matA(1000, 1000).tocsc()
    print("GET MATA")
    kernel_yr = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    kernel_yl = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    kernel_xu = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    kernel_xd = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])

    matCXU = scipy.signal.convolve2d(source,
                                     kernel_xu,
                                     boundary='symm',
                                     mode='same').flatten()
    matCXD = scipy.signal.convolve2d(source,
                                     kernel_xd,
                                     boundary='symm',
                                     mode='same').flatten()
    matDXU = scipy.signal.convolve2d(target,
                                     kernel_xu,
                                     boundary='symm',
                                     mode='same').flatten()
    matDXD = scipy.signal.convolve2d(target,
                                     kernel_xd,
                                     boundary='symm',
                                     mode='same').flatten()

    matCYL = scipy.signal.convolve2d(source,
                                     kernel_yl,
                                     boundary='symm',
                                     mode='same').flatten()
    matCYR = scipy.signal.convolve2d(source,
                                     kernel_yr,
                                     boundary='symm',
                                     mode='same').flatten()
    matDYL = scipy.signal.convolve2d(target,
                                     kernel_yl,
                                     boundary='symm',
                                     mode='same').flatten()
    matDYR = scipy.signal.convolve2d(target,
                                     kernel_yr,
                                     boundary='symm',
                                     mode='same').flatten()

    matBXU = matCXU
    matBYR = matCYR
    matBXD = matCXD
    matBYL = matCYL
    for i in range(matBXU.shape[0]):
        if (abs(matDXU[i]) > abs(matCXU[i])):
            matBXU[i] = matDXU[i]
        if (abs(matDXD[i]) > abs(matCXD[i])):
            matBXD[i] = matDXD[i]
        if (abs(matDYR[i]) > abs(matCYR[i])):
            matBYR[i] = matDYR[i]
        if (abs(matDYL[i]) > abs(matCYL[i])):
            matBYL[i] = matDYL[i]

    matB = matBXU + matBXD + matBYR + matBYL
    print(matB.shape)
    #matB = matB.flatten()
    target = target.flatten()
    print("EDITING")
    for x in range(1000):
        matB[0 * 1000 + x] = target[0 * 1000 + x]
        matB[x * 1000] = target[1000 * x]
        matB[x * 1000 + 999] = target[x * 1000 + 999]
        matB[1000 * 1000 - x - 1] = target[1000 * 1000 - x - 1]

    print("SOLVING")
    X = sc.sparse.linalg.spsolve(matA, matB)
    X[X > 255] = 255
    X[X < 0] = 0
    X = X.reshape((1000, 1000))
    tar[1000:2000, 1000:2000] = X
    print("ALMOST")
    return tar


(src_r, src_g, src_b) = cv.split(source)
(tar_r, tar_g, tar_b) = cv.split(target)

result = cv.merge((edit(src_r, tar_r), edit(src_g, tar_g), edit(src_b, tar_b)))

cv.imwrite("mixed_result_4.png", result)
