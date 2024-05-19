import cv2
import sys, os, time
import numpy as np

input_filepath  = ""
if len(sys.argv) >= 2:
        input_filepath = sys.argv[1]
else:
    print("No input image specified.")


output_filepath  = "extracted.png"
if len(sys.argv) >= 2:
        output_filepath = sys.argv[2]
else:
    print("No output filepath specified. Resulting image will be saved as \"extracted.png\"")

WIDTH = 100
if len(sys.argv) >= 4:
    try:
        WIDTH = int(sys.argv[3])
    except ValueError:
        print("Given width is not a number")
else:
    "No width specified. Width has been set to 100."
    
HEIGHT = 100
if len(sys.argv) >= 5:
    try:
        HEIGHT = int(sys.argv[4])
    except ValueError:
        print("Given height is not a number")
else:
    "No height specified. Height has been set to 100."

base_img = np.zeros((50,50))
try:
    base_img = cv2.imread(input_filepath)
except:
    print(f"Could not read image {input_filepath}")
    os._exit(1)

WINDOW_NAME = 'Extractor Window'
RESULT_WINDOW_NAME = 'Preview Window'

cv2.namedWindow(WINDOW_NAME)

img = base_img.copy()

points:np.array = np.zeros(shape=(4, 2))
num_recorded_points:int = 0

def mouse_callback(event, x, y, flags, param):
    global img
    global num_recorded_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if num_recorded_points < 4:
            points[num_recorded_points] = np.array((x,y))
            num_recorded_points += 1
            img = cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(WINDOW_NAME, img)
        if num_recorded_points >= 4:
            time.sleep(1)
            cv2.destroyWindow(WINDOW_NAME)
            sort_points()
            show_result()

def reset_points():
    global img
    global points
    global num_recorded_points
    img = base_img.copy()

    points = np.zeros(shape=(4, 2))
    num_recorded_points = 0


def to_polar(point:np.array, origin:np.array=np.array((0,0))) -> np.array:
    p_relative = point-origin
    theta = np.arctan2(p_relative[1], p_relative[0])
    r = np.sqrt(p_relative[0]**2 + p_relative[1]**2)
    return np.array((r, theta))

def to_cartesian(point:np.array, origin:np.array):
    r = point[0]
    theta = point[1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    p_relative = np.array((x,y))
    return p_relative + origin

def show_result():
    global base_img
    global points
    homography = calculate_homography()
    result_img = cv2.warpPerspective(base_img, homography, (WIDTH, HEIGHT))
    cv2.imshow(RESULT_WINDOW_NAME, result_img)
    while True:
        key = cv2.waitKey()
        if key == ord('a'):
            points = np.roll(points, -1, axis=0)
            show_result()
            break
        elif key == ord('d'):
            points = np.roll(points, 1, axis=0)
            show_result()
            break
        elif key == 27:
            cv2.destroyWindow(RESULT_WINDOW_NAME)
            reset_points()
            show_extractor()
            break
        elif key == 13 or key == ord('s'):
            save_image(result_img)
            break
    


def save_image(img:np.array):
    global output_filepath
    cv2.imwrite(output_filepath, img)
    print(f"Image saved to {output_filepath}.")
    os._exit(0)




def sort_points():
    global points
    origin = np.mean(points,0)
    polar = np.apply_along_axis(lambda a : to_polar(a, origin), 1, points)
    polar = polar[polar[:, 1].argsort()] # https://stackoverflow.com/a/2828121
    points = np.apply_along_axis(lambda a : to_cartesian(a, origin), 1, polar)
    


def calculate_homography():
    big_img_cornerpoints = np.array([[0     , 0],
                                     [WIDTH , 0],
                                     [WIDTH , HEIGHT],
                                     [0     , HEIGHT]])
    homography, ret = cv2.findHomography(points,big_img_cornerpoints)
    return homography
    


def show_extractor ():
    cv2.imshow(WINDOW_NAME, img)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

show_extractor()
cv2.waitKey()


