
import cv2 as cv
import numpy as np

BLUR_PARAMETER = 3

def get_kernel(kernel_size, blur_parameter):
    ker = np.zeros((kernel_size, kernel_size), dtype=float)
    a = kernel_size - kernel_size // 2
    b = a
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i+1, j+1
            val = (1 / (2* np.pi * blur_parameter**2)) * np.exp(-(((x-a)**2 + (y-b)**2)/(2 * blur_parameter**2)))
            ker[i, j] = val
    
    return ker

kernel3 = get_kernel(3, BLUR_PARAMETER)
kernel5 = get_kernel(5, BLUR_PARAMETER)
kernel7 = get_kernel(7, BLUR_PARAMETER)
"""
print(f"РАЗМЕР 3x3:\n{kernel3}")
print(f"РАЗМЕР 5x5:\n{kernel5}")
print(f"РАЗМЕР 7x7:\n{kernel7}")
"""

kernel3 = kernel3 / kernel3.sum()
kernel5 = kernel5 / kernel5.sum()
kernel7 = kernel7 / kernel7.sum()

"""
print(f"Нормированная матрица размера 3x3:\n{kernel3}\nСумма элементов: {kernel3.sum()}")
print(f"Нормированная матрица размера 5x5:\n{kernel5}\nСумма элементов: {kernel5.sum()}")
print(f"Нормированная матрица размера 7x7:\n{kernel7}\nСумма элементов: {kernel7.sum()}")
"""
# Пункт 3, 4, 5
frame = cv.imread("input.png")

grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

cv.imwrite("original_gray.png", grayscale_frame)

kernel_sizes = [3, 9]
blur_params = [2, 10]

for ks in kernel_sizes:
    for bp in blur_params:
        kernel = get_kernel(ks, bp)
        kernel = kernel / kernel.sum()  

        print(f"Нормированная матрица:\n{kernel}\nСумма элементов: {kernel.sum()}")

        h, w = grayscale_frame.shape
        pad = ks // 2

        padded = cv.copyMakeBorder(grayscale_frame, pad, pad, pad, pad, cv.BORDER_REFLECT)

        blurred_frame = np.zeros_like(grayscale_frame, dtype=float)
        for i in range(h):
            for j in range(w):
                region = padded[i:i + ks, j:j + ks]
                value = np.sum(region * kernel)
                blurred_frame[i, j] = value

        blurred_frame = np.clip(blurred_frame, 0, 255).astype(np.uint8)
        filename = f"blur_k{ks}_b{bp}.png"
        cv.imwrite(filename, blurred_frame)

        blurred_cv = cv.GaussianBlur(grayscale_frame, (ks, ks), bp)
        cv_filename = f"blur_cv_k{ks}_b{bp}.png"
        cv.imwrite(cv_filename, blurred_cv)
