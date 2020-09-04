import cv2
import numpy as np

minimum_size = 6000  # Area of the contours detected


def pre_processing(frame):
    # Use HSV image format
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 70], np.uint8)

    # Get the object in between red_lower and red_upper only
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    kernel = np.ones((5, 5), "uint8")

    # Image Processing Morphology to make the object larger
    black_mask = cv2.dilate(black_mask, kernel)

    contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_details = {}
    bounding_box_area = []
    bounding_box = []
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # print(area)

        # Only take the object bigger than minimum size, remove noise
        if area > minimum_size:  # 10000
            detect = True
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # print("aspect ratio", aspect_ratio)

            # Make sure the aspect ratio of width and height is squared
            if 0.95 < aspect_ratio < 1.1:
                # Label the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Black Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

                # Save details in python dictionary
                bounding_box_area.append(area)
                bounding_box.append([x, y, w, h])

    contour_details = {
        "area": bounding_box_area,
        "bounding_box": bounding_box
    }
    # cv2.imshow('black mask', black_mask)
    # cv2.imshow('hsv frame', hsvFrame)
    # cv2.imshow('original', frame)
    # cv2.waitKey(0)

    return contour_details