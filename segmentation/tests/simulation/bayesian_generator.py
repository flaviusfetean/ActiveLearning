import os.path

import numpy as np
import cv2
import random

from scipy.special import softmax


def generate_binary_image(width, height, num_circles):
    # Create a white background image
    image = np.ones((height, width), dtype=np.uint8) * 255

    circles = []

    # Generate random circles
    while len(circles) < num_circles:
        # Random position
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)

        # Random size
        radius = random.randint(5, min(width, height) // 4)

        # Check for overlap with previous circles
        overlapping = False
        for circle in circles:
            circle_center_x, circle_center_y, circle_radius = circle
            distance = np.sqrt((circle_center_x - center_x) ** 2 + (circle_center_y - center_y) ** 2)
            if distance < circle_radius + radius:
                overlapping = True
                break

        # If no overlap, add the circle to the list
        if not overlapping:
            circles.append((center_x, center_y, radius))
            cv2.circle(image, (center_x, center_y), radius, 0, -1)

    return image


def generate_binary_images(width, height, num_circles, num_images, overlap_probability):
    images = []

    for _ in range(num_images):
        # Create a white background image
        image = np.ones((height, width), dtype=np.uint8) * 255

        circles = []

        # Generate random circles
        while True:
            # Random position
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)

            # Random size
            radius = random.randint(5, min(width, height) // 4)

            # Check for overlap with previous circles
            overlapping = False
            for circle in circles:
                circle_center_x, circle_center_y, circle_radius, color = circle
                distance = np.sqrt((circle_center_x - center_x) ** 2 + (circle_center_y - center_y) ** 2)
                if distance < circle_radius + radius:
                    overlapping = True
                    break

            # If no overlap or if random chance allows overlap, add the circle to the list
            if not overlapping or random.random() < overlap_probability:
                circles.append((center_x, center_y, radius))
                cv2.circle(image, (center_x, center_y), radius, color, -1)

            # Exit the loop if we have generated enough circles
            if len(circles) == num_circles:
                break

        images.append(image)

    return images


def generate_base_image(width, height, num_circles, circle_sizes):
    frame_minsize = min(width, height)

    min_size, max_size = int(circle_sizes[0] * frame_minsize), int(circle_sizes[1] * frame_minsize)
    # Create a white background image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    circles = []

    # Generate non-overlapping random circles for the base image
    for _ in range(num_circles):
        while True:
            # Random position
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)

            # Random size
            radius = random.randint(min_size, max_size)

            #Random color
            color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])  # Red, Green, Blue

            # Check for overlap with previous circles
            overlapping = False
            for circle in circles:
                circle_center_x, circle_center_y, circle_radius, _ = circle
                distance = np.sqrt((circle_center_x - center_x)**2 + (circle_center_y - center_y)**2)
                if distance < circle_radius + radius:
                    overlapping = True
                    break

            # If no overlap, add the circle to the list
            if not overlapping:
                circles.append((center_x, center_y, radius, color))
                cv2.circle(image, (center_x, center_y), radius, color, -1)
                break

    return image, circles


def generate_random_overlaps(width, height, num_circles, circle_sizes, num_images, stddev):
    images = []
    previous_image, previous_circles = generate_base_image(width, height, circle_sizes, num_circles)

    images.append(previous_image)

    for _ in range(1, num_images):
        # Create a white background image for the new image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        new_circles = []

        # Generate new circles based on previous image
        for circle in previous_circles:
            prev_center_x, prev_center_y, prev_radius, color = circle

            # Sample new center from Gaussian distribution
            new_center_x = int(np.random.normal(prev_center_x, stddev))
            new_center_y = int(np.random.normal(prev_center_y, stddev))

            # Calculate new radius based on overlap probability
            new_radius = prev_radius

            # Add the new circle to the list
            if new_radius > 2:
                new_circles.append((new_center_x, new_center_y, new_radius))
                cv2.circle(image, (new_center_x, new_center_y), new_radius, color, -1)

        images.append(image)

    return images


def categorical_to_probability(image, num_classes):
    h, w = image.shape[:2]

    sim_prediction = np.random.rand(h * w * num_classes)
    sim_prediction = sim_prediction.reshape((h, w, num_classes))
    sim_prediction = softmax(sim_prediction, axis=-1)
    sim_labels = np.argmax(np.expand_dims(image, axis=-1), axis=-1)
    crt_labels = np.argmax(sim_prediction, axis=-1)

    j, i = np.meshgrid(range(w), range(h))
    sim_prediction[i, j, sim_labels], sim_prediction[i, j, crt_labels] = \
        sim_prediction[i, j,  crt_labels], sim_prediction[i, j, sim_labels]

    return sim_prediction


def convert_to_probability(images):
    num_classes = len(np.unique(images))
    sim_predictions = np.array([categorical_to_probability(image, num_classes) for image in images])

    return sim_predictions


def generate_probability_outputs(width, height, num_circles, circle_sizes, num_images, stddev):
    images = generate_random_overlaps(width, height, num_circles, circle_sizes, num_images, stddev)

    return convert_to_probability(images)


def aggregate_outputs(images, show=False):
    accumulator = np.average(np.array(images), axis=0)

    if show:
        cv2.imshow("Combined Image", accumulator.astype("uint8"))

    return accumulator


def main3():
    SAVE_PATH = "/segmentation\\tests\\simulation\\outputs_samples\\color_samples\\random2"
    # Example usage
    width = 256
    height = 256
    num_images = 10
    num_circles = 6
    circles_sizes = (0.01, 0.3)
    stddev = 12

    binary_images = generate_random_overlaps(width, height, circles_sizes, num_circles, num_images, stddev)

    for i, image in enumerate(binary_images):
        cv2.imwrite(os.path.sep.join([SAVE_PATH, f"output_{i}.png"]), image)

    aggregate_outputs(binary_images, show=True)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main2():
    # Example usage
    width = 500
    height = 500
    num_images = 3
    overlap_probability = 0.5

    binary_images = generate_binary_images(width, height, 4, num_images, overlap_probability)

    # Display the images

    RGB_image = np.dstack(binary_images)
    cv2.imshow("Overlapped imgs", RGB_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Example usage
    width = 500
    height = 500
    num_circles = 4

    binary_image = generate_binary_image(width, height, num_circles)

    # Display the image
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main3()


