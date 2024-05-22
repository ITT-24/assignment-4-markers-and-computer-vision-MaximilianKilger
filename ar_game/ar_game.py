import cv2
import numpy as np
import pyglet
import sys,os
from cv2 import aruco
from PIL import Image
import random
from threading import Thread
import math


WIDTH = 100
HEIGHT = 100

CAT_FILEPATH = os.path.join("assets", "cat.png")
TIGER_FILEPATH = os.path.join("assets", "tiger.png")

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)
area_corners = np.array([[-1,-1],
                         [-1,-1],
                         [-1,-1],
                         [-1,-1]]) # dummy points for homography

def extract_area (frame:np.array, area_corners:np.array ):
    WIDTH = frame.shape[1]
    HEIGHT = frame.shape[0]
    big_img_cornerpoints = np.array([[0     , 0],
                                    [WIDTH , 0],
                                    [WIDTH , HEIGHT],
                                    [0     , HEIGHT]])

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(gray)
    # Check if marker is detected
    if ids is not None:
        # Draw lines along the sides of the marker
        aruco.drawDetectedMarkers(frame, corners)
        if len(ids) == 4:

            corners = np.array(corners)[ids.flatten().argsort()]
            for i, corner in enumerate(corners):
                area_corners[i] = corner[0][(i+2)%4]

    if not -1 in area_corners:
        homography, ret = cv2.findHomography(area_corners,big_img_cornerpoints)
        result_img = cv2.warpPerspective(frame, homography, (WIDTH, HEIGHT))
        return True,result_img
    else:
        return False, cv2.resize(frame, (WIDTH, HEIGHT))

def get_center_of_hand(frame:np.array):

    #thresh = cv2.adaptiveThreshold(result_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 121, 2)
    #ret, thresh = cv2.threshold(result_img, 120, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    s = cv2.GaussianBlur(s, (11,11), 0)
    ret, s_thresh = cv2.threshold(s, 38, 255, cv2.THRESH_BINARY)
    #contours, hierarchy = cv2.findContours(s_thresh, 1, 2)
    dist = cv2.distanceTransform(s_thresh,cv2.DIST_L2,5)
    max_dist = np.max(dist)
    index = None
    if max_dist > WIDTH*0.085:
        index  = np.unravel_index(dist.argmax(), dist.shape)[::-1]
    dist = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)
    
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    #if not index is None:
    #    dist = cv2.circle(dist, index, 15, (0,0,255))
        
    #cv2.imshow("ASDF", dist)
    return index

# code von Andi oder Tina
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

class GameManager ():
    def __init__(self):
        self.cat_mode:str = "CAT"
        self.cursor_pos:tuple = (-1,-1)
        self.cursor_distance:int = 0
        self.REQUIRED_CURSOR_DISTANCE:int = 200
        self.TIGER_PROVOCATION_DISTANCE:int = 0.25 * self.REQUIRED_CURSOR_DISTANCE
        self.CAT_CHANCE = 0.8
        self.status:str = "PRE_GAME"
        self.score:int = 0
        self.ui_setup = False
        self.ROUND_LENGTH = 5

        self.CAT_IMG = pyglet.image.load(CAT_FILEPATH)
        self.TIGER_IMG = pyglet.image.load(TIGER_FILEPATH)
        
    def init_ui (self):
        self.ui_setup = True
        self.BAR_LENGTH = 200
        self.BAR_HEIGHT = 20
        self.BAR_VERT_DISTANCE = 30

        self.EMPTY_COLOR = (128,128,128, 255)
        self.FULL_COLOR = (222, 80, 84)

        self.CAT_WIDTH = window.width * 0.2
        self.CAT_HEIGHT = self.CAT_WIDTH * 2

        self.CAT_UPPER_STOPPING_POINT = window.height - 100
        self.CAT_LOWER_STOPPING_POINT = -100
        self.CAT_SPEED = 3


        self.bar_background = pyglet.shapes.Rectangle((window.width - self.BAR_LENGTH)/2, window.height - (self.BAR_VERT_DISTANCE+self.BAR_HEIGHT), self.BAR_LENGTH, self.BAR_HEIGHT)
        self.bar = pyglet.shapes.Rectangle((window.width - self.BAR_LENGTH)/2, self.BAR_VERT_DISTANCE+self.BAR_HEIGHT, 0, self.BAR_HEIGHT)
        self.cat_sprite:pyglet.sprite.Sprite = pyglet.sprite.Sprite(self.CAT_IMG, (window.width-self.CAT_WIDTH) / 2, self.CAT_LOWER_STOPPING_POINT - self.CAT_HEIGHT, 0)
        self.cat_sprite.height = self.CAT_HEIGHT
        self.cat_sprite.width = self.CAT_WIDTH

    def update_ui(self):
        print(self.status)
        if self.ui_setup:
            self.bar.width = min(self.cursor_distance/self.REQUIRED_CURSOR_DISTANCE, 1) * self.BAR_LENGTH

            if self.status == "RAISING":
                projected_y_pos = self.cat_sprite.y + self.CAT_SPEED()
                if projected_y_pos + self.cat_sprite.height >= self.CAT_UPPER_STOPPING_POINT:
                    self.cat_sprite.y = min(projected_y_pos+self.cat_sprite.height, self.CAT_UPPER_STOPPING_POINT)
                    self.status = 'ACTIVE'
                    pyglet.clock.schedule_once(self.resolve_round, self.ROUND_LENGTH)
            if self.status == "ACTIVE":
                if self.cat_mode == "CAT":
                    if self.cursor_distance >= self.REQUIRED_CURSOR_DISTANCE:
                        self.resolve_round()
                        pyglet.clock.unschedule(self.resolve_round)
                elif self.cat_mode == "TIGER":
                    if self.cursor_distance >= self.TIGER_PROVOCATION_DISTANCE:
                        self.resolve_round()
                        pyglet.clock.unschedule(self.resolve_round)
            if self.status == "LOWERING":
                projected_y_pos = self.cat_sprite.y + self.CAT_SPEED()
                if projected_y_pos + self.cat_sprite.height <= self.CAT_LOWER_STOPPING_POINT:
                    self.cat_sprite.y = max(projected_y_pos+self.cat_sprite.height, self.CAT_LOWER_STOPPING_POINT)
                    self.choose_cat()
                    self.status = 'RAISING'
            if self.status == "PRE_GAME":
                self.status == "RAISING"

            
    def choose_cat(self):
        if random.random() <= self.cat_chance:
            self.cat_mode = "CAT"
        else:
            self.cat_mode = "TIGER"

    def resolve_round(self):
        if self.status == 'ACTIVE':
            if self.cat_mode == "CAT":
                if self.cursor_distance >= self.REQUIRED_CURSOR_DISTANCE:
                    self.score += 1
            elif self.cat_mode == "TIGER":
                if self.cursor_distance >= self.TIGER_PROVOCATION_DISTANCE:
                    self.score -= 1
            self.cursor_distance = 0
            self.status = "LOWERING"
                

            
        

    def update_cursor_position(self, new_pos:tuple):
        if new_pos is None:
            return
        x_dist = self.cursor_pos[0] - new_pos[0]
        y_dist = self.cursor_pos[1] -  new_pos[1]
        dist = math.sqrt(x_dist**2 + y_dist**2)
        self.cursor_pos = new_pos
        if self.status == "ACTIVE":
            if self.cursor_on_cat():
                self.cursor_distance += dist
    
        
    def cursor_on_cat(self):
        if self.cat_sprite is None:
            return False
        left_bound = self.cat_sprite.x
        right_bound = self.cat_sprite.x + self.cat_sprite.width
        lower_bound = self.cat_sprite.y
        upper_bound = self.cat_sprite.y + self.cat_sprite.height
        xpos, ypos = self.cursor_pos
        within_x_bounds = left_bound < xpos and xpos < right_bound
        within_y_bounds = lower_bound < ypos and ypos < upper_bound
        return within_x_bounds and within_y_bounds

    def render(self):
        if self.ui_setup:
            if not self.cat_sprite is None:
                self.cat_sprite.draw()
            self.bar_background.draw()
            self.bar.draw()

window = pyglet.window.Window(WIDTH, HEIGHT)
gameManager = GameManager()
IMG = cv2glet(np.zeros((10,10)), "GRAY")

marker_pos = (-20000,-20000)
marker = pyglet.shapes.Circle(marker_pos[0], marker_pos[1], 20, color=(255,25,0, 255))


@window.event
def on_draw():
    if window.height != HEIGHT or window.width != WIDTH:
        window.set_size(WIDTH, HEIGHT)
        gameManager.init_ui()
    gameManager.update_ui()
    window.clear()
    IMG.blit(0,0,0)
    gameManager.render()
    marker.x = marker_pos[0]
    marker.y = HEIGHT-marker_pos[1]
    marker.draw()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:
        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
        os._exit(0)

def main_loop():
    global IMG
    global WIDTH
    global HEIGHT
    global marker_pos
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if ret == True and not frame is None:
            HEIGHT, WIDTH = frame.shape[:2]
            ret, result_img = extract_area(frame, area_corners)
            hand_position = get_center_of_hand(result_img)
            IMG = cv2glet(result_img,"BGR")
            if not hand_position is None:
                marker_pos = hand_position

img_read_thread = Thread(target=main_loop)
img_read_thread.start()

pyglet.app.run()


