import cv2
import numpy as np


def read_image(image_path):
    return cv2.imread(image_path)


def apply_gaussian_blur(image, kernel_size=(15, 15)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_canny_edge_detection(gray_image, threshold1=100, threshold2=200):
    return cv2.Canny(gray_image, threshold1, threshold2)

def apply_dilation(edges, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(edges, kernel, iterations)

def apply_erosion(edges, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(edges, kernel, iterations)

def apply_opening(edges, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

def apply_closing(edges, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

def apply_gradient(edges, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

def manual_dilation_single_patch(roi, kernel):
    roi_height, roi_width = roi.shape
    kernel_height, kernel_width = kernel.shape
    dilated_roi = np.zeros_like(roi)

    for i in range(roi_height):
        for j in range(roi_width):
            y_min = max(i - kernel_height // 2, 0)
            y_max = min(i + kernel_height // 2 + 1, roi_height)
            x_min = max(j - kernel_width // 2, 0)
            x_max = min(j + kernel_width // 2 + 1, roi_width)
            
            region = roi[y_min:y_max, x_min:x_max]
            kernel_region = kernel[y_min-i + kernel_height//2 : y_max-i + kernel_height//2, 
                                   x_min-j + kernel_width//2 : x_max-j + kernel_width//2]
            
            if np.any(region & kernel_region):
                dilated_roi[i, j] = 255

    return dilated_roi

def apply_manual_dilation(edges, kernel):
    roi = edges[50:60, 50:60]
    dilated_roi = manual_dilation_single_patch(roi, kernel)
    edges[50:60, 50:60] = dilated_roi

def bitwise_operations(image, edge_image):
    bitwise_and = cv2.bitwise_and(image, image, mask=edge_image) 
    bitwise_or = cv2.bitwise_or(image, image, mask=edge_image) 
    bitwise_xor = cv2.bitwise_xor(image, image, mask=edge_image)
    return bitwise_and, bitwise_or, bitwise_xor

def display_images(images):
    for title, img in images.items():
        cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image_path):
    image = read_image(image_path)
    gaussian_blur = apply_gaussian_blur(image)
    gray_image = convert_to_gray(image)
    
    canny_image = apply_canny_edge_detection(gray_image)
    
    dilated_image = apply_dilation(canny_image)
    eroded_image = apply_erosion(canny_image)
    opening = apply_opening(canny_image)
    closing = apply_closing(canny_image)
    gradient = apply_gradient(canny_image)

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    apply_manual_dilation(canny_image, kernel)
    bitwise_and, bitwise_or, bitwise_xor = bitwise_operations(image, canny_image)
    
    images = {
        'Original Image': image,
        'Gaussian Blur': gaussian_blur,
        'Canny Edges': canny_image,
        'eroded_image':eroded_image,
        'Dilated Edges': dilated_image,
        'Opening': opening,
        'Closing': closing,
        'Gradient': gradient,
        'Bitwise AND': bitwise_and,
        'Bitwise OR': bitwise_or,
        'Bitwise XOR': bitwise_xor
    }
    display_images(images)

if __name__ == "__main__":
    main('bird2.jpg')






























