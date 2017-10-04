# beeline.py
#
# This program displays the webcam in a window. The user can drag the mouse to
# draw a line. Then the program will use background subtraction to find moving
# (foreground) objects. We'll scan along the line and calculate the average of
# all forgeground pixels, giving us a position along the line.
#
# Other keys:
#   'q' or ESC to quit
#   'b' to toggle between the webcam image and the foreground/background
#       separation
#
# Version 1.0, 2017-09-17.
#
# Created using:
#   Python 3.6.2
#     https://www.python.org/downloads/release/python-362/
#   Numpy 1.13.1
#     https://www.scipy.org/scipylib/download.html
#     or "pip3 install numpy"
#   OpenCV 3.3.0
#     https://pypi.python.org/pypi/opencv-python
#     or "pip3 install opencv-python"

import numpy as np
import cv2

# Adapted from:
#   https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
def create_line_iterator(p1, p2, img):
    """
    Produces an array that consists of the coordinates and intensities of each
    pixel in a line between two points.

    Parameters:
        - p1: a numpy array with the coordinate of the first point (x,y)
        - p2: a numpy array with the coordinate of the second point (x,y)
        - img: the image being processed

    Returns:
        - it: a numpy array that consists of the coordinates and intensities
          of each pixel in the radii (shape: [numPixels, 3],
          row = [x,y,intensity])
    """

    # Define local variables for readability.
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = np.array(p1[0])
    P1Y = np.array(p1[1])
    P2X = np.array(p2[0])
    P2Y = np.array(p2[1])

    # Difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # Predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: # vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: # horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    # Remove points outside of image.
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    # Get intensities from img ndarray.
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),
                        itbuffer[:,0].astype(np.uint)]

    return itbuffer

# A class to hold the current line and provide a mouse callback.
class Line:
    def __init__(self):
        # Start and end of the line segment.
        self.start_point = np.array((0, 0))
        self.end_point = np.array((0, 0))
        # Are we in the middle of a mouse drag?
        self.button_down = False
        # Has the line been made?
        self.have_line = False
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point[0] = x
            self.start_point[1] = y
            self.button_down = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.button_down:
                self.end_point[0] = x
                self.end_point[1] = y
                self.have_line = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.button_down = False
            
    def draw(self, image):
        if self.have_line:
            # Draw a blue line.
            cv2.line(image, (self.start_point[0], self.start_point[1]),
                     (self.end_point[0], self.end_point[1]), (255, 0, 0), 2)
            
    def find_object_on_line(self, image, fg_mask):
        if self.button_down or not self.have_line:
            return
        average_point = np.array((0, 0))
        count = 0
        it = create_line_iterator(self.start_point, self.end_point, fg_mask)
        for v in it:
            # If the mask has a value over 128, then we'll consider it a
            # foreground pixel, and accumulate its location into average_point.
            if v[2] > 128:
                average_point += np.array((int(v[0]), int(v[1])))
                count += 1
        # If we had enough samples, then find and draw the average point.
        if count > 4:
            average_point //= count
            cv2.circle(image, (int(average_point[0]), int(average_point[1])),
                       4, (0, 0, 255), -1)
            # Calculate the distance from the starting point. The ratio of this
            # distance to the line's length tells us how far along the line
            # we are, on a scale of 0...1.
            dist_from_start = np.linalg.norm(average_point - self.start_point)
            proportional_distance = (dist_from_start /
                np.linalg.norm(self.end_point - self.start_point))
            # Scale from 0...1 to 0...10.
            value_to_print = proportional_distance * 10
            cv2.putText(image, "%.2f" % (value_to_print), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

def main():
    line = Line()
    show_separation = False

    # Start capturing from device 0.
    cap = cv2.VideoCapture(0)
    background_subtractor = cv2.createBackgroundSubtractorKNN()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", line.mouse_callback)

    while(True):
        # Capture frame-by-frame.
        ret, frame = cap.read()
        if not ret:
            print("Can't capture a frame.")
            break

        foreground_mask = background_subtractor.apply(frame)
        if show_separation:
            frame = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2RGB)
        line.draw(frame)
        line.find_object_on_line(frame, foreground_mask)

        # Display the resulting frame
        cv2.imshow("image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # 'q' or ESC to quit
            break
        elif key == ord('b'):
            show_separation = not show_separation

    # When everything is done, release the capture.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()