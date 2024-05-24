## Image Extraction
Call the script extract_image.py from the command line as such:
python3 extract_image.py [input filepath] [output filepath] [output width] [output height]
All parameters except input filepath are optional and have default values assigned to them.

In the Extractor view, you will see the input image. Click on the corners of the desired output image to select them.

In the Preview view, you can use the following keys to interact:

**A** to rotate the image clockwise. (This will not change the aspect ratio.)

**D** to rotate the image counterclockwise.

**ESC** to return to the Extractor view and make adjustments.

**S** or **ENTER** to save the image and exit the program.

## AR Game
The objective of the AR game is to pet the cat. Only pet the cat. Don't pet the tiger.
The AR game was tested under controlled lighting conditions. When you play the game, the lighting conditions may be different. If the detection doesn't work right, try to eliminate light from outside. If that doesn't help, try playing around with the CV constants (especially SATURATION_THRESH).
