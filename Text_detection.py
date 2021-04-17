import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'

import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt


def apply_filter(image):
    """
    Applying 5*5 kernel filter filtering
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Image')
    # plt.show()
    return filtered


def apply_threshold(filtered):
    """
    Apply binary or adaptive threshold (optional)
    Args:
        filtered: np.array
    Returns:
        thresh: np.array
    """
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 199, 15)
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    plt.title('After applying binary thresholding')
    # plt.show()
    return thresh


def contour(img, image_shape):
    """
    Args:
        img: np.array()
        image_shape: tuple
    Returns:
        canvas: np.array()
        cnt: list
    """
    canvas = np.zeros(image_shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
    plt.title('Largest Contour')
    plt.imshow(canvas)
    # plt.show()

    return canvas, cnt


def detect_corners_from_contour(canvas, cnt):
    """
    Detecting corner points form contours using cv2.approxPolyDP()
    Args:
        canvas: np.array()
        cnt: list
    Returns:
        approx_corners: list
    """
    epsilon = 0.03 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    print("approx_corners:", approx_corners)

    # Rearranging the order of the corner points
    approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]
    plt.imshow(canvas)
    plt.title('Image Countor')
    # plt.show()
    return approx_corners


def corner_points(corners):
    """
    -Get destination points from corners of image
    -Approximating height and width of a image
    Args:
        corners: list
    Returns:
        destination_corners: list
        height: int
        width: int
    """

    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    # print('\nApproximated height and width of the image : \n', (h, w))
    return destination_corners, h, w


def unwarp(img, src, dst):
    """
    Args:
        img: np.array
        src: list
        dst: list
    Returns:
        un_warped: np.array
    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    # print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    # plot

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.imshow(img)
    ax1.set_title('Input Image')

    x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
    y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]

    ax2.imshow(img)
    ax2.plot(x, y, color='Blue', linewidth=3)
    ax2.set_ylim([h, 0])
    ax2.set_xlim([0, w])
    ax2.set_title('Detected Boundary Area')

    plt.show()
    return un_warped


def example_two():
    """
    Skew correction using homography and corner detection using contour points
    Returns: None
    """
    image = cv2.imread('t2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Input Image')
    plt.show()

    filtered_image = apply_filter(image)
    threshold_image = apply_threshold(filtered_image)

    cnv, largest_contour = contour(threshold_image, image.shape)
    corners = detect_corners_from_contour(cnv, largest_contour)

    destination_points, h, w = corner_points(corners)
    un_warped = unwarp(image, np.float32(corners), destination_points)

    cropped = un_warped[0:h, 0:w]
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # f.subplots_adjust(hspace=.2, wspace=.05)
    # ax1.imshow(un_warped)
    plt.figure(figsize=(13, 7))
    plt.imshow(cropped)
    plt.title('Transformed Image')
    plt.show()

    # pytesseract text detection
    text1 = pytesseract.image_to_string(cropped)
    print("FETCHED TEXT FROM THE IMAGE: ")

    print(text1)


if __name__ == '__main__':
    example_two()