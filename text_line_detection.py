import os

import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def anisotropic_gaussian_filter(ksize_x, ksize_y, sigma_x, sigma_y):
    # Generate meshgrid
    x, y = np.meshgrid(np.arange(-ksize_x // 2 + 1, ksize_x // 2 + 1),
                       np.arange(-ksize_y // 2 + 1, ksize_y // 2 + 1))

    # Compute Gaussian components
    gaussian_x = np.exp(-x ** 2 / (2 * sigma_x ** 2))
    gaussian_y = np.exp(-y ** 2 / (2 * sigma_y ** 2))

    # Compute anisotropic Gaussian filter
    filter_kernel = gaussian_x * gaussian_y

    # Normalize the filter
    filter_kernel /= np.sum(filter_kernel)

    return filter_kernel


def extract_text_and_heights(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Get bounding boxes for individual components (words or lines)
        # You may need to adjust the config parameter based on your image and OCR requirements
        # Assuming you already have a function to obtain bounding boxes
        component_boxes = get_bounding_boxes(img)
        # Calculate heights of components
        component_heights = [box[3] - box[1] for box in component_boxes]
    return component_heights


def get_bounding_boxes(image):
    # Convert image to grayscale
    grayscale_image = image.convert('L')

    # Convert PIL image to NumPy array of type np.uint8
    img_array = np.array(grayscale_image)

    # Perform thresholding to create binary image
    _, binary_image = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to extract bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))  # Format: (x1, y1, x2, y2)

    return bounding_boxes


def calculate_mean_and_std(heights):
    return np.mean(heights), np.std(heights)


def calculate_character_height_range(mean, std):
    return mean, mean + std


def get_mean_std(image_path):
    componet_height = extract_text_and_heights(image_path)
    if len(componet_height) == 1:
        return componet_height[0], 1
    return calculate_mean_and_std(componet_height)
    # height_range = calculate_character_height_range(mean, std)


def get_blob_lines(filtered_image):
    # Find contours in the filtered image
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area or other criteria
    blob_lines = [contour for contour in contours if cv2.contourArea(contour) > 100]  # Adjust threshold as needed

    return blob_lines


def calculate_fs1_score(component, max_character_height):
    # Calculate the average 1-norm of each pixel from the component
    # You may need to extract the coordinates of the pixels within the component
    # For simplicity, let's assume a placeholder calculation
    average_1_norm = np.random.uniform(0, 10)  # Placeholder value

    # Check if the average 1-norm satisfies the condition
    if average_1_norm < max_character_height:
        return True
    else:
        return False


def calculate_fs2_score(component, blob_area):
    # Calculate the ratio of the blob area to the sum of the distances of the contour pixels from the spline
    # You may need to calculate the distances of the contour pixels from the spline
    # For simplicity, let's assume a placeholder calculation
    ratio = np.random.uniform(0, 1)  # Placeholder value

    # Check if the ratio satisfies the condition
    if ratio < 0.9:
        return True
    else:
        return False


def get_blob_line_image(image_path, elongation_rate, ksize_x, threshold):
    # Load the original image
    image_file_path = image_path
    file_name = "text_line_extraction"
    path_to_save_image = f"/Users/omerbensalmon/Desktop/BGU/Semester_5/mini_project_CV/final_product/test_resault/{file_name}/elongation_{elongation_rate}_ksizex_{ksize_x}_threshold{threshold}"
    path_to_save_image_contur_image = f"{path_to_save_image}__lines_image.jpg"
    # path_to_save_image_contur_image_with_description = f"{path_to_save_image}__lines_image_with_description.jpg"
    path_to_save_image_bubble_image = f"{path_to_save_image}__gausian_filter_image.jpg"
    # path_to_save_image_bubble_image_with_description = f"{path_to_save_image}__gausian_filter_image_with_description.jpg"
    text_original = f"elongation_rate = {elongation_rate}, ksize_x = {ksize_x}, threshold = {threshold}"

    original_image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

    # Given values
    mean_height, std_deviation = get_mean_std(image_file_path)  # Average height of components
    # elongation_rate = 2  # Optimal elongation rate

    # Calculate sigma_x
    sigma_x = std_deviation
    if sigma_x == 0:
        return

    # Calculate sigma_y using elongation rate
    sigma_y = elongation_rate * sigma_x
    x = list(range(1, 220, 20))
    y = list(range(1, 100, 5))

    # Parameters for anisotropic Gaussian filter
    # ksize_x = 300
    ksize_y = 20

    # Create the anisotropic Gaussian filter
    filter_kernel = anisotropic_gaussian_filter(ksize_x, ksize_y, sigma_x, sigma_y)

    # Apply the filter to the original image
    filtered_image = cv2.filter2D(original_image, -1, filter_kernel)
    filtered_image = 255 - filtered_image

    _, bubble_image = cv2.threshold(filtered_image, threshold, 255, cv2.THRESH_BINARY)

    blob_lines = get_blob_lines(bubble_image)

    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGRA)
    # bubble_image = cv2.cvtColor(bubble_image, cv2.COLOR_GRAY2BGRA)

    for line in blob_lines:
        # Create a mask for the contour
        cv2.drawContours(original_image, [line], -1, (0, 255, 255), 4)
    # Get the dimensions of the image

    cv2.imshow("Contour Image", original_image)
    # cv2.imshow("Bubble Image", bubble_image)
    cv2.waitKey(0)  # Wait for any key press to close the windows
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == '__main__':
    elongation_rate = 2
    ksize_x = 1
    threshold = 70
    image_path = os.path.join(os.getcwd(), 'support_data/image_1.jpg')
    get_blob_line_image(image_path, elongation_rate, ksize_x, threshold)
