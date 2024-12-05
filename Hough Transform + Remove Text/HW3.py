import cv2 as cv
import numpy as np

image = cv.imread('road_lines2.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 50, 150, apertureSize=3)

standard_hough_image = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  
probabilistic_hough_image = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

# Standard Hough Transform
lines = cv.HoughLines(edges, rho=1, theta=np.pi/180, threshold=200)

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(standard_hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines

# Probabilistic Hough Transform
min_line_length = 50  # Minimum length of line to be considered
max_line_gap = 10     # Maximum allowed gap between line segments

p_lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                          minLineLength=min_line_length, maxLineGap=max_line_gap)

if p_lines is not None:
    for x1, y1, x2, y2 in p_lines[:, 0]:
        cv.line(probabilistic_hough_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  

cv.imshow('Standard Hough Transform on Edges', standard_hough_image)
cv.imshow('Probabilistic Hough Transform on Edges', probabilistic_hough_image)

cv.waitKey(0)
cv.destroyAllWindows()

#############################################################

image = cv.imread('cat1.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

_, mask = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

# Inpainting (removing the text)
result = cv.inpaint(image, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

cv.imshow('Original Image', image)
cv.imshow('Image after Inpainting', result)

cv.imwrite('image_without_text.jpg', result)

cv.waitKey(0)
cv.destroyAllWindows()

###############################################

image = cv.imread('bankai.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

_, mask = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

# Find contours of the text
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

text_mask = np.zeros_like(mask)

# Filling the detected contours (areas where text is) with white
cv.drawContours(text_mask, contours, -1, (255), thickness=cv.FILLED)

# an inpainted version of the image where the text is removed
result_inpainted = cv.inpaint(image, text_mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

cv.imshow('Original Image', image)
cv.imshow('Text Mask', text_mask)
cv.imshow('Inpainted Image', result_inpainted)

cv.imwrite('text_mask.jpg', text_mask)
cv.imwrite('image_without_text.jpg', result_inpainted)

cv.waitKey(0)
cv.destroyAllWindows()