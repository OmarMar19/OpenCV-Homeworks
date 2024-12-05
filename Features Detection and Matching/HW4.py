import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(image_paths):
    images = [cv2.imread(path) for path in image_paths]
    for i, img in enumerate(images):
        if img is None:
            print(f"Error: Unable to load image {image_paths[i]}")
            exit(1)
    return images

def select_region_of_interest(image):
    """Let the user select a region of interest (ROI) on the image."""
    print("Select the object you want to match, and press Enter or Space to confirm.")
    roi = cv2.selectROI("Select ROI", image)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = map(int, roi)
    return image[y:y+h, x:x+w]

def match_orb_bf(template_image, target_image):
    
    orb = cv2.ORB_create()
    kp_template, des_template = orb.detectAndCompute(template_image, None)
    kp_target, des_target = orb.detectAndCompute(target_image, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_template, des_target)
    matches = sorted(matches, key=lambda x: x.distance)
    
    result = cv2.drawMatches(template_image, kp_template, target_image, kp_target, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result

def match_sift_flann(template_image, target_image):
    
    sift = cv2.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(template_image, None)
    kp_target, des_target = sift.detectAndCompute(target_image, None)
    
    index_params = dict(algorithm=1, trees=5)  # FLANN parameters
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_template, des_target, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    result = cv2.drawMatches(template_image, kp_template, target_image, kp_target, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result

def match_sift_bf(template_image, target_image):
    """SIFT with Brute Force Matching."""
    sift = cv2.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(template_image, None)
    kp_target, des_target = sift.detectAndCompute(target_image, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_template, des_target)
    matches = sorted(matches, key=lambda x: x.distance)
    
    result = cv2.drawMatches(template_image, kp_template, target_image, kp_target, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result

def match_template(template_image, target_image):
    """Template Matching."""
    gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    res = cv2.matchTemplate(gray_target, gray_template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    h, w = gray_template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    result = target_image.copy()
    cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 2)
    return result

def process_images(template_image, target_images):

    sift = cv2.SIFT_create()  

    
    kp_template, des_template = sift.detectAndCompute(cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY), None)
    
    
    template_with_keypoints = cv2.drawKeypoints(template_image, kp_template, None, color=(0, 255, 0))
    cv2.imshow("Template Keypoints", template_with_keypoints)
    cv2.waitKey(0)
    """Perform different types of matching and display results."""
    fig, axs = plt.subplots(len(target_images), 4, figsize=(20, 5 * len(target_images)))
    fig.suptitle("Feature Matching Results", fontsize=16)
    
    for i, target_image in enumerate(target_images):
        orb_result = match_orb_bf(template_image, target_image)
        sift_flann_result = match_sift_flann(template_image, target_image)
        sift_bf_result = match_sift_bf(template_image, target_image)
        template_result = match_template(template_image, target_image)
        
        axs[i, 0].imshow(cv2.cvtColor(orb_result, cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title("ORB + BF Matching")
        axs[i, 0].axis("off")
        
        axs[i, 1].imshow(cv2.cvtColor(sift_flann_result, cv2.COLOR_BGR2RGB))
        axs[i, 1].set_title("SIFT + FLANN Matching")
        axs[i, 1].axis("off")
        
        axs[i, 2].imshow(cv2.cvtColor(sift_bf_result, cv2.COLOR_BGR2RGB))
        axs[i, 2].set_title("SIFT + BF Matching")
        axs[i, 2].axis("off")
        
        axs[i, 3].imshow(cv2.cvtColor(template_result, cv2.COLOR_BGR2RGB))
        axs[i, 3].set_title("Template Matching")
        axs[i, 3].axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Paths to images
    image_paths = [
        "C:/Users/Samer/Desktop/5th Year/Vision/HW4/img1.jpg", 
        "C:/Users/Samer/Desktop/5th Year/Vision/HW4/img2.jpg", 
        "C:/Users/Samer/Desktop/5th Year/Vision/HW4/img3.jpg"
    ]  
    images = load_images(image_paths)
    
   
    print("Select the object in the first image.")
    template_image = select_region_of_interest(images[0])
    
   
    process_images(template_image, images[1:])
