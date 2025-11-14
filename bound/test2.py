import cv2 as cv
import numpy as np

def discretize_direction(gx, gy, tg):
    
    dir_map = np.zeros_like(gx, dtype=np.uint8)

    m0 = ((gx > 0) & (gy < 0) & (tg < -2.414)) | ((gx < 0) & (gy < 0) & (tg > 2.414))
    dir_map[m0] = 0

    m1 = (gx > 0) & (gy < 0) & (tg >= -2.414) & (tg < -0.414)
    dir_map[m1] = 1

    m2 = ((gx > 0) & (gy < 0) & (tg >= -0.414)) | ((gx > 0) & (gy > 0) & (tg < 0.414))
    dir_map[m2] = 2

    m3 = (gx > 0) & (gy > 0) & (tg >= 0.414) & (tg < 2.414)
    dir_map[m3] = 3

    m4 = ((gx > 0) & (gy > 0) & (tg >= 2.414)) | ((gx < 0) & (gy > 0) & (tg <= -2.414))
    dir_map[m4] = 4

    m5 = (gx < 0) & (gy > 0) & (tg > -2.414) & (tg <= -0.414)
    dir_map[m5] = 5

    m6 = ((gx < 0) & (gy > 0) & (tg > -0.414)) | ((gx < 0) & (gy < 0) & (tg < 0.414))
    dir_map[m6] = 6

    m7 = (gx < 0) & (gy < 0) & (tg >= 0.414) & (tg < 2.414)
    dir_map[m7] = 7

    return dir_map


def run_edge_detection(path_to_file):
    
    src_img = cv.imread(path_to_file)
    if src_img is None:
        print(f"Ошибка: файл '{path_to_file}' не открыт")
        return

    
    gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    smooth_img = cv.GaussianBlur(gray_img, (3, 3), 2)

  
    kernel_sobel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float64)
    kernel_sobel_y = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]], dtype=np.float64)

    img_f = smooth_img.astype(np.float64)

    gx = np.zeros_like(img_f)
    gy = np.zeros_like(img_f)

    
    pad_img = np.pad(img_f, ((1, 1), (1, 1)), mode='reflect')

    
    for r in range(img_f.shape[0]):
        for c in range(img_f.shape[1]):
            window = pad_img[r:r+3, c:c+3]
            gx[r, c] = np.sum(window * kernel_sobel_x)
            gy[r, c] = np.sum(window * kernel_sobel_y)

    
    grad_mag = np.sqrt(gx**2 + gy**2)

    
    gx_safe = np.where(gx == 0, 1e-6, gx)
    tang = gy / gx_safe

    
    dirs = discretize_direction(gx, gy, tang)

    
    nms_mask = np.zeros_like(grad_mag, dtype=np.uint8)
    H, W = grad_mag.shape

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            d = dirs[r, c]
            val = grad_mag[r, c]

            if d in [0, 4]:
                neigh = [grad_mag[r, c-1], grad_mag[r, c+1]]
            elif d in [1, 5]:
                neigh = [grad_mag[r-1, c+1], grad_mag[r+1, c-1]]
            elif d in [2, 6]:
                neigh = [grad_mag[r-1, c], grad_mag[r+1, c]]
            else:
                neigh = [grad_mag[r-1, c-1], grad_mag[r+1, c+1]]

            if val > max(neigh):
                nms_mask[r, c] = 255
            else:
                nms_mask[r, c] = 0

    
    max_val = np.max(grad_mag)
    high_thr = max_val // 7
    low_thr = max_val // 13

    strong = (grad_mag >= high_thr)
    weak = ((grad_mag >= low_thr) & (grad_mag < high_thr))

    out = np.zeros_like(nms_mask, dtype=np.uint8)
    out[strong & (nms_mask == 255)] = 255

    
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if weak[r, c] and nms_mask[r, c] == 255:
                neigh_box = out[r-1:r+2, c-1:c+2]
                if np.any(neigh_box == 255):
                    out[r, c] = 255

    
    overlay_img = src_img.copy()
    overlay_img[out == 255] = [0, 0, 255]  

    
    cv.imshow("Source", src_img)
    cv.imshow("Gray + Smooth", smooth_img)
    cv.imshow("NMS", nms_mask)
    cv.imshow("Double Threshold", out)
    cv.imshow("Edges Overlay", overlay_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return overlay_img




run_edge_detection("input4.png")
