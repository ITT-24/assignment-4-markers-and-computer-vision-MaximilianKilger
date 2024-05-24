import cv2
import numpy as np
import pyglet
import sys,os
from cv2 import aruco
from PIL import Image
import random
from threading import Thread
import math
import time


WIDTH = 100
HEIGHT = 100

CAT_FILEPATHS = [os.path.join("assets", "cat.png"),
                os.path.join("assets", "cat2.png"),
                os.path.join("assets", "cat3.png"),
                os.path.join("assets", "cat4.png")]
TIGER_FILEPATH = os.path.join("assets", "tiger.png")
HAND_FILEPATH = os.path.join("assets", "hand.png")

# computer vision parameters. If you are having trouble with detection in your current environment, playing around with these might help.
SATURATION_THRESH = 38
BLUR_RADIUS = 9
MIN_HAND_POINT_DISTANCE = 0.085


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

def extract_area (frame:np.array, area_corners:np.array )->tuple[bool,np.array]:
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
        # correct image
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
    s = cv2.GaussianBlur(s, (BLUR_RADIUS,BLUR_RADIUS), 0)
    ret, s_thresh = cv2.threshold(s, SATURATION_THRESH, 255, cv2.THRESH_BINARY)
    if ret:
        #contours, hierarchy = cv2.findContours(s_thresh, 1, 2)
        dist = cv2.distanceTransform(s_thresh,cv2.DIST_L2,5)
        max_dist = np.max(dist)
        index = None
        if max_dist > WIDTH*MIN_HAND_POINT_DISTANCE:
            index  = np.unravel_index(dist.argmax(), dist.shape)[::-1]
        #dist = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)
        
        #cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        #if not index is None:
        #    dist = cv2.circle(dist, index, 15, (0,0,255))
            
        #cv2.imshow("ASDF", dist)
        return index
    else:
        return None

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
        self.REQUIRED_CURSOR_DISTANCE:int = 3000
        self.TIGER_PROVOCATION_DISTANCE:int = 0.2 * self.REQUIRED_CURSOR_DISTANCE
        self.CAT_CHANCE = 0.75
        self.status:str = "PRE_GAME"
        self.score:int = 0
        self.ui_setup = False
        self.DEFAULT_ROUND_LENGTH = 6
        self.round_length = 6
        self.ROUND_LENGTH_ACCELERATION = 0.1
        self.round_timer:float = 0
        self.GAME_OVER_TIME = 5

        self.CAT_IMGS = []
        for fp in CAT_FILEPATHS:
            self.CAT_IMGS.append(pyglet.image.load(fp))
        self.TIGER_IMG = pyglet.image.load(TIGER_FILEPATH)
        self.HAND_IMG = pyglet.image.load(HAND_FILEPATH)
        
    def init_ui (self):
        self.ui_setup = True
        self.BAR_LENGTH = 200
        self.BAR_HEIGHT = 20
        self.BAR_VERT_DISTANCE = 30

        self.EMPTY_COLOR = (128,128,128, 255)
        self.FULL_COLOR = (222, 80, 84)
        self.TIME_COLOR = (18, 220, 0)

        self.CAT_WIDTH = window.width * 0.5
        self.CAT_HEIGHT = self.CAT_WIDTH * 2

        self.CAT_UPPER_STOPPING_POINT = window.height - 100
        self.CAT_LOWER_STOPPING_POINT = -100
        self.CAT_SPEED = 10

        self.SCORE_LABEL_SIZE = 15
        self.SCORE_LABEL_MARGIN = 20

        self.SCORE_SIZE = 30
        self.SCORE_COLOR = (0,0,192,255)

        self.TIME_BAR_HEIGHT = 5
        self.TIME_BAR_VERT_DISTANCE = 10

        self.HAND_WIDTH = 256


        self.bar_background = pyglet.shapes.Rectangle((window.width - self.BAR_LENGTH)/2, window.height - (self.BAR_VERT_DISTANCE+self.BAR_HEIGHT), self.BAR_LENGTH, self.BAR_HEIGHT, color=self.EMPTY_COLOR)
        self.bar = pyglet.shapes.Rectangle((window.width - self.BAR_LENGTH)/2, window.height-(self.BAR_VERT_DISTANCE+self.BAR_HEIGHT), 0, self.BAR_HEIGHT, color=self.FULL_COLOR)
        self.time_bar_background = pyglet.shapes.Rectangle((window.width - self.BAR_LENGTH)/2, window.height - (self.BAR_VERT_DISTANCE+self.BAR_HEIGHT+self.TIME_BAR_VERT_DISTANCE+self.TIME_BAR_HEIGHT), self.BAR_LENGTH, self.TIME_BAR_HEIGHT, color=self.EMPTY_COLOR)
        self.time_bar = pyglet.shapes.Rectangle((window.width - self.BAR_LENGTH)/2,  window.height - (self.BAR_VERT_DISTANCE+self.BAR_HEIGHT+self.TIME_BAR_VERT_DISTANCE+self.TIME_BAR_HEIGHT), 0, self.TIME_BAR_HEIGHT, color=self.TIME_COLOR)
        self.cat_sprite:pyglet.sprite.Sprite = pyglet.sprite.Sprite(self.CAT_IMGS[0], (window.width-self.CAT_WIDTH) / 2, self.CAT_LOWER_STOPPING_POINT - self.CAT_HEIGHT, 0)
        self.cat_sprite.height = self.CAT_HEIGHT
        self.cat_sprite.width = self.CAT_WIDTH
        
        self.hand_sprite:pyglet.sprite.Sprite = pyglet.sprite.Sprite(self.HAND_IMG, x=-2000, y=-2000)
        self.hand_sprite.height = self.HAND_WIDTH
        self.hand_sprite.width = self.HAND_WIDTH
        self.hand_sprite

        self.score_text_label = pyglet.text.Label("SCORE", "Arial", self.SCORE_LABEL_SIZE,color=self.SCORE_COLOR, x=self.SCORE_LABEL_MARGIN, y=window.height-self.SCORE_LABEL_MARGIN)
        self.score_label = pyglet.text.Label("", "Arial", self.SCORE_SIZE, bold=True, color=self.SCORE_COLOR, x=self.SCORE_LABEL_MARGIN, y = window.height - (2*self.SCORE_LABEL_MARGIN+self.SCORE_LABEL_SIZE))

    def update(self):
        #print(self.status)
        if self.ui_setup:
            cursor_x, cursor_y = self.cursor_pos
            self.hand_sprite.x = cursor_x - self.HAND_WIDTH/2
            self.hand_sprite.y = window.height-(cursor_y + self.HAND_WIDTH/2)
            if self.status != "GAME_OVER":
                self.bar.width = min(self.cursor_distance/self.REQUIRED_CURSOR_DISTANCE, 1) * self.BAR_LENGTH
                self.time_bar.width = min(self.round_timer/self.round_length, 1) * self.BAR_LENGTH
                self.score_label.text = str(self.score)
            if self.status == "PRE_GAME":
                self.status = "RAISING"
            if self.status == "RAISING":
                projected_y_pos = self.cat_sprite.y + self.CAT_SPEED
                if projected_y_pos + self.cat_sprite.height >= self.CAT_UPPER_STOPPING_POINT:
                    self.cat_sprite.y = min(projected_y_pos, self.CAT_UPPER_STOPPING_POINT-self.cat_sprite.height)
                    self.round_timer = 0
                    self.delta_time = time.time()
                    self.status = 'ACTIVE'
                    #pyglet.clock.schedule_once(self.resolve_round, self.round_length)
                else:
                    self.cat_sprite.y = projected_y_pos
            if self.status == "ACTIVE":
                print(f"{self.cursor_distance}")
                now = time.time()
                self.round_timer += now-self.delta_time
                self.delta_time = now
                if self.cat_mode == "CAT":
                    if self.cursor_distance >= self.REQUIRED_CURSOR_DISTANCE:
                        self.resolve_round()
                        return
                        #pyglet.clock.unschedule(self.resolve_round)
                elif self.cat_mode == "TIGER":
                    if self.cursor_distance >= self.TIGER_PROVOCATION_DISTANCE:
                        self.game_over()
                        return
                        #pyglet.clock.unschedule(self.resolve_round)
                if self.round_timer >= self.round_length:
                    self.resolve_round()
                
                
            if self.status == "LOWERING":
                projected_y_pos = self.cat_sprite.y - self.CAT_SPEED
                if projected_y_pos + self.cat_sprite.height <= self.CAT_LOWER_STOPPING_POINT:
                    self.cat_sprite.y = max(projected_y_pos, self.CAT_LOWER_STOPPING_POINT-self.cat_sprite.height)
                    self.choose_cat()
                    self.status = 'RAISING'
                else:
                    self.cat_sprite.y = projected_y_pos
    
    def game_over(self):
        self.status = "GAME_OVER"
        self.score_text_label.text="GAME OVER"
        self.score_label.text = ":("
        pyglet.clock.schedule_once(self.reset, self.GAME_OVER_TIME)

    def reset(self, dt=0):
        self.round_timer = 0
        self.cursor_distance = 0
        self.score = 0
        self.cat_sprite.y = self.CAT_LOWER_STOPPING_POINT - self.CAT_HEIGHT
        self.round_length = self.DEFAULT_ROUND_LENGTH
        self.choose_cat()
        self.status = "RAISING"



            
    def choose_cat(self):
        if random.random() <= self.CAT_CHANCE:
            self.cat_mode = "CAT"
            cat_index = random.randrange(0,len(self.CAT_IMGS))
            self.cat_sprite.image = self.CAT_IMGS[cat_index]
            self.cat_sprite.width = self.CAT_WIDTH
            self.cat_sprite.height = self.CAT_HEIGHT
        else:
            self.cat_mode = "TIGER"
            self.cat_sprite.image = self.TIGER_IMG
            self.cat_sprite.width = self.CAT_WIDTH
            self.cat_sprite.height = self.CAT_HEIGHT

    def resolve_round(self, dt=None):
        if self.status == 'ACTIVE':
            if self.cat_mode == "CAT":
                if self.cursor_distance >= self.REQUIRED_CURSOR_DISTANCE:
                    self.score += 1
                    self.round_length -= self.ROUND_LENGTH_ACCELERATION
            self.cursor_distance = 0
            self.status = "LOWERING"
            self.round_timer = 0

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
            self.hand_sprite.draw()
            self.bar_background.draw()
            self.bar.draw()
            self.time_bar_background.draw()
            self.time_bar.draw()
            self.score_text_label.draw()
            self.score_label.draw()


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
    gameManager.update()
    window.clear()
    IMG.blit(0,0,0)
    gameManager.render()

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
                gameManager.update_cursor_position(hand_position)

img_read_thread = Thread(target=main_loop)
img_read_thread.start()

pyglet.app.run()


