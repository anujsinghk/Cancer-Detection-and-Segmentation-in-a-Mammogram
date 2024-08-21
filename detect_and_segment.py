import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import closing, square
from skimage.filters import sobel

def apply_averaging_filter(image, filter_size):
    """Apply an averaging filter to the image."""
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size ** 2)
    smoothed_image = convolve(image, kernel)
    return smoothed_image

def find_patch_region(image, thresholded_image):
    """Identify the patch region that contains the bright pixels."""
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour corresponds to the cancerous region
    max_contour = max(contours, key=cv2.contourArea)
    
    # Create a bounding rectangle around the largest contour
    x, y, w, h = cv2.boundingRect(max_contour)
           
    return (x, y, w, h)

def threshold_image(image, threshold_value):
    """Apply thresholding to separate cancerous regions."""
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

def max_mean_least_variance(image, sub_window_size, X, Y):
    """Detect cancerous region using Max-Mean and Least-Variance technique."""
    rows, cols = image.shape
    detected_region = np.zeros_like(image)
    
    sub_windows = []
    for i in range(0, rows - sub_window_size + 1, sub_window_size):
        for j in range(0, cols - sub_window_size + 1, sub_window_size):
            sub_window = image[i:i + sub_window_size, j:j + sub_window_size]
            mean_intensity = np.mean(sub_window)
            variance = np.var(sub_window)
            sub_windows.append((mean_intensity, variance, i, j))
    
    # Sort sub-windows by mean intensity (descending)
    sub_windows = sorted(sub_windows, key=lambda x: x[0], reverse=True)
    
    # Select the top X sub-windows by mean intensity
    top_mean_sub_windows = sub_windows[:X]
    
    # From these, select the top Y sub-windows by least variance
    top_mean_sub_windows = sorted(top_mean_sub_windows, key=lambda x: x[1])
    top_variance_sub_windows = top_mean_sub_windows[:Y]
    
    # Highlight the detected region in the image
    for _, _, i, j in top_variance_sub_windows:
        detected_region[i:i + sub_window_size, j:j + sub_window_size] = 255
    
    return detected_region

def apply_morphological_closing(image, size):
    """Apply morphological closing to smooth the edges and fill small holes."""
    return closing(image, square(size))

def apply_image_gradient(image):
    """Apply gradient to highlight the edges."""
    return sobel(image)

def detect_and_segment(image, filter_size=25, threshold_value=125, sub_window_size=3, M=2800, V=1800):
    """Main function to detect and segment cancerous regions in a mammogram image."""
    smoothed_image = apply_averaging_filter(image, filter_size)
    thresholded_image = threshold_image(smoothed_image, threshold_value)
    
    # Assuming rectangular window surrounds the patch
    x, y, w, h = find_patch_region(image, thresholded_image)
    
    # Extract the patch region from the original image
    patch_region = image[y:y+h, x:x+w]
    
    # Apply Max-Mean and Least Variance technique on the rectangular window
    highlighted_region = max_mean_least_variance(patch_region, sub_window_size, M, V)
    
    # replacing in original image position 
    detected_region = np.zeros_like(image)
    detected_region[y:y+h, x:x+w] = highlighted_region


    # Apply morphological closing and image gradient
    closed_image = apply_morphological_closing(detected_region, size=10)
    gradient_image = apply_image_gradient(closed_image)
    
    # highlight detection tumor region in original image
    tumor_region = image
    tumor_region[gradient_image > 0] = 255
    cv2.imwrite("./output/tumor_region.png", tumor_region)


    # Display the intermediate and final results
    cv2.imshow(f"Original Image", image)
    cv2.imshow(f"Thresholded Image {threshold_value}", thresholded_image)
    cv2.imshow(f"Detected Cancer Region", detected_region)
    cv2.imshow(f"Closed Image", closed_image)
    cv2.imshow(f"Gradient Image", gradient_image)
    cv2.imshow(f"Tumor_region", image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
# Load the mammogram image (grayscale)
image = cv2.imread('./input/mammogram.png', 0)

# Detect and segment cancerous regions
detect_and_segment(image,  sub_window_size=3, M=2800, V=1800)
