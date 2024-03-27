import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables for images
im1 = None
im2 = None
im3 = cv2.imread('florence3.jpg')

def get_pixel_coordinates(image_path_1, image_path_2, num_clicks=5):
    '''
    Function to interactively get pixel coordinates from the user for two images.
    This function allows the user to select 5 points on two images. After selecting 5 points in an image,
    it does not allow you to select more points on the same image.
    It writes each selected point to the console, specifying which point it is, and at the end of the process,
    it shows full version, including array states.

    Right click select!(event.button==3 means right click on the mouse)

    Args:
    - image_path_1: Path to the first image
    - image_path_2: Path to the second image
    - num_clicks: Number of clicks required from the user
    Returns:
    - chosen_points_1: Pixel coordinates for florence1
    - chosen_points_2: Pixel coordinates for florence2
    '''
    global im1, im2
    im1 = cv2.imread(image_path_1)
    im2 = cv2.imread(image_path_2)

    # Create subplots for the two images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    implot1 = ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
    implot2 = ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))

    ax1.set_title('florence1')
    ax2.set_title('florence2')

    # Lists to store clicked points for each image
    clicked_points_1 = []
    clicked_points_2 = []

    def onclick(event):
        # Event handler for mouse clicks
        nonlocal clicked_points_1, clicked_points_2

        if event.xdata is not None and event.ydata is not None:
            # Check if right mouse button is clicked on florence1 and florence2
            if event.inaxes == ax1 and event.button == 3 and len(clicked_points_1) < num_clicks:
                clicked_points_1.append((event.xdata, event.ydata))
                print("Clicked Point for florence1: ({:.2f}, {:.2f})".format(event.xdata, event.ydata))
            elif event.inaxes == ax2 and event.button == 3 and len(clicked_points_2) < num_clicks:
                clicked_points_2.append((event.xdata, event.ydata))
                print("Clicked Point for florence2: ({:.2f}, {:.2f})".format(event.xdata, event.ydata))

        if len(clicked_points_1) == num_clicks and len(clicked_points_2) == num_clicks:
            plt.close()
    # Connect the event handler
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    # Convert the clicked points to NumPy arrays
    chosen_points_1 = np.array(clicked_points_1)
    chosen_points_2 = np.array(clicked_points_2)

    return chosen_points_1, chosen_points_2

def draw_epipolar_lines_and_intersection_points(im1, im2, im3, points1, points2, Fmatrix13, Fmatrix23):
    '''
    Function to draw epipolar lines and intersection points on three images.
    This function draws the points and epipolar lines where the selected points on Florence1 and Florence2 coincide
    on Florence3 and uses fundamental matrices for this.

    Fmatrix13 for florence1 and florence3
    Fmatrix23 for florence2 and florence3
    Args:
    - im1(florence1), im2(florence3), im3(florence3): Images
    - points1, points2: Corresponding pixel coordinates in im1 and im2
    - Fmatrix13, Fmatrix23: Fundamental matrices
    '''
    # Colors for points in florence1 and florence2
    colors_im1 = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    colors_im2 = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    # Normalize colors to the range of 0-1
    normalized_colors_im1 = np.array(colors_im1) / 255.0
    normalized_colors_im2 = np.array(colors_im2) / 255.0

    # Plot images and overlay selected points
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))

    # Overlay selected points on im1 with normalized colors
    [ax1.scatter(x, y, c=[color], marker='o') for (x, y), color in zip(points1, normalized_colors_im1)]
    ax1.set_title('florence1 With Points')

    # Compute epipolar lines in florence3 using Fmatrix13 for points in florence1
    points_homogeneous_im1 = np.hstack((points1, np.ones((len(points1), 1))))
    epipolar_lines_im3_im1 = np.dot(Fmatrix13, points_homogeneous_im1.T).T

    # Plot epipolar lines on florence3 with normalized colors from florence1
    [ax3.plot((-b * np.linspace(0, im3.shape[0], num=100) - c) / a, np.linspace(0, im3.shape[0], num=100),
              color=color, linewidth=0.8) for (a, b, c), color in zip(epipolar_lines_im3_im1, normalized_colors_im1)]

    # Now, repeat the process for points in florence2
    ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    # Overlay selected points on florence2 with normalized colors
    [ax2.scatter(x, y, c=[color], marker='o') for (x, y), color in zip(points2, normalized_colors_im2)]
    ax2.set_title('florence2 With Points')

    # Compute epipolar lines in florence3 using Fmatrix23 for points in florence2
    points_homogeneous_im2 = np.hstack((points2, np.ones((len(points2), 1))))
    epipolar_lines_im3_im2 = np.dot(Fmatrix23, points_homogeneous_im2.T).T

    # Plot epipolar lines on florence3 with normalized colors from florence2
    [ax3.plot((-b * np.linspace(0, im3.shape[0], num=100) - c) / a, np.linspace(0, im3.shape[0], num=100),
              color=color, linewidth=0.8) for (a, b, c), color in zip(epipolar_lines_im3_im2, normalized_colors_im2)]
    # Compute and plot intersection points
    intersection_points = [np.linalg.solve(np.array([[line1[0], line1[1]], [line2[0], line2[1]]]),
                                           np.array([-line1[2], -line2[2]])) for line1, line2 in
                           zip(epipolar_lines_im3_im1, epipolar_lines_im3_im2)]
    # Plot intersection points after plotting epipolar lines
    [ax3.scatter(point[0], point[1], color=color, marker='o', s=50) for point, color in
     zip(intersection_points, normalized_colors_im1)]
    ax3.imshow(cv2.cvtColor(im3, cv2.COLOR_BGR2RGB))
    ax3.set_title('florence3 with Epipolar Lines and Intersection Points')
    plt.show()


if __name__ == "__main__":
    # Get pixel coordinates for chosen points
    image_path_1 = "florence1.jpg"
    image_path_2 = "florence2.jpg"
    chosen_points_1, chosen_points_2 = get_pixel_coordinates(image_path_1, image_path_2, num_clicks=5)
    # Print the selected points for Image 1 and Image 2
    print("Chosen Points for florence1:\n", chosen_points_1)
    print("Chosen Points for florence2:\n", chosen_points_2)
    # Fundamental matrices
    Fmatrix13 = np.array([
        [6.04444985855117e-08, 2.56726410274219e-07, -0.000602529673152695],
        [2.45555247713476e-07, -8.38811736871429e-08, -0.000750892330636890],
        [-0.000444464396704832, 0.000390321707113558, 0.999999361609429]
    ])
    Fmatrix23 = np.array([
        [3.03994528999160e-08, 2.65672654114295e-07, -0.000870550254997210],
        [4.67606901933558e-08, -1.11709498607089e-07, -0.00169128012255720],
        [-1.38310618285550e-06, 0.00140690091935593, 0.999997201170569]
    ])
    # Draw epipolar lines and intersection points
    draw_epipolar_lines_and_intersection_points(im1, im2, im3, chosen_points_1, chosen_points_2, Fmatrix13, Fmatrix23)