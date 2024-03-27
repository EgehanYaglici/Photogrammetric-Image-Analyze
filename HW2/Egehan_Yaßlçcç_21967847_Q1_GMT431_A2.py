import cv2
import numpy as np
import matplotlib.pyplot as plt


def pixel_coordinate_selection(imagePath, num_clicks=5):
    '''
    Gets pixel coordinates from user clicks on the image.

    This function opens a matplotlib window for you and stores the pixel coordinates of each place you press with
    the right button of your mouse in an array list. You can see the coordinates of each place you press in the console.
    It allows you to select only 5 points for the purpose of the assignment.

    Right click selects! (event.button==3 means right click on the mouse)

    Parameters:
    - imagePath: str, path to the image file
    - num_clicks: int, number of clicks to capture

    Returns:
    - chosen_points: np.ndarray, array of selected pixel coordinates
    '''
    # Read the image using matplotlib
    im = plt.imread(imagePath)

    # Create a plot and display the image
    fig, ax = plt.subplots()
    ax.set_title("florence2")
    implot = ax.imshow(im)

    # Store clicked points
    clicked_points = []

    def onclick(event):
        nonlocal clicked_points
        # Check if a valid point is clicked with the right mouse button
        if event.xdata is not None and event.ydata is not None and event.button == 3:
            clicked_points.append((event.xdata, event.ydata))
            print("Clicked Point: ({:.2f}, {:.2f})".format(event.xdata, event.ydata))

            # If the required number of clicks is reached, close the plot
            if len(clicked_points) == num_clicks:
                plt.close()

    # Connect the click event to the defined function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Display the plot
    plt.show()

    # Convert the clicked points to a NumPy array
    chosen_points = np.array(clicked_points)

    return chosen_points


def plot_points_and_epipolar_lines(im1, im2, F, points):
    '''
    Plots points and epipolar lines on two images given fundamental matrix and selected points.

    This function multiplies our fundamental matrix with the pixel coordinates we choose and determines the epipolar
    lines (dot product). It processes the matrix by transposing the points and transposes the result, thus the lines
    are determined. In the remaining part, he draws the points on the florence2 and the lines on the florence3 image.

    Parameters:
    - im1: np.ndarray, first image(florence2)
    - im2: np.ndarray, second image(florence3)
    - F: np.ndarray, fundamental matrix
    - points: np.ndarray, array of selected pixel coordinates
    '''
    # Add homogeneous coordinates to points
    points_h = np.hstack((points, np.ones((5, 1))))

    # Compute epipolar lines with the fundamental matrix
    lines = np.dot(F, points_h.T).T

    # Plot images and points
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    ax1.imshow(im1)
    ax1.set_title("florence2")

    ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    ax2.set_title("florence3")

    # Overlay points on florence2
    [ax1.scatter(point[0], point[1], c=f'C{i}', marker='o') for i, point in enumerate(points)]

    # Overlay epipolar lines on florence3
    [ax2.plot(np.array([0, im2.shape[1]]), - (line[0] * np.array([0, im2.shape[1]]) + line[2]) / line[1],
              f'C{i}', label=f"Point {i + 1}") for i, line in enumerate(lines)]

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Specify the path of the first image
    image_path_1 = "florence2.jpg"

    # Get user-selected pixel coordinates
    chosen_points = pixel_coordinate_selection(image_path_1, num_clicks=5)

    # Fundamental matrix
    F = np.array([[3.03994528999160e-08, 2.65672654114295e-07, -0.000870550254997210],
                  [4.67606901933558e-08, -1.11709498607089e-07, -0.00169128012255720],
                  [-1.38310618285550e-06, 0.00140690091935593, 0.999997201170569]])

    # Read the second image
    im2 = cv2.imread('florence3.jpg')

    # Plot points and epipolar lines
    plot_points_and_epipolar_lines(im1=plt.imread(image_path_1), im2=im2, F=F, points=chosen_points)
