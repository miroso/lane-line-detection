[//]: # (Image References)

[image0]: ./output_images/output_image_text.png "Cover"
[image1]: ./output_images/undistorted_image.png "Undistorted c"
[image2]: ./output_images/undistorted_image_2.png "Undistorted 2"
[image3]: ./output_images/binary_image_steps.png "Gradient and Color"
[image4]: ./output_images/warped_image_roi.png "Warped"
[image5]: ./output_images/hist.png "Histogram"
[image6]: ./output_images/sliding_window.png "Sliding Window"
[image7]: ./output_images/detected_lines.png "Detected Lines"
[image8]: ./output_images/output_image.png "Unwarped"
[image9]: ./output_images/output_image_text.png "Text"
[video1]: ./project_video.mp4 "Video"

### Lane Line Detection
![Cover Image][image0]

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. 

Programming Language: Python
Method: Sliding Window
Notes: The IPython notebook contains a compact summary of all steps taken in this project. 


This README will address the following points:

1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
2. Provide an example of a distortion-corrected image.
3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
4. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
5. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
6. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
7. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
8. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
9. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

___

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for the camera calibration is contained in **step 1**. In order to compute the camera calibration matrix and the distortion coefficients, we first need to get the image and object points. This is done by the function `get_image_and_object_points()`, which takes chessboard calibration images and the number of the inside chessboard corners along the x and y axis as input parameters. For each calibration image the function finds the chessboard corners using `cv2.findChessboardCorners()` and appends them to a set of image points. The object points represent the chessboard corners in the real world. Assuming that the chessboard is fixed on the (x, y) plane at z=0, then the object points are the same for each calibration image and will be replicated for each successful detection of image points. The object and image points are then fed to `get_camera_calibration_parameters()`, in which the `cv2.calibrateCamera()` function computes the camera calibration and distortion coefficients.

The camera calibration and distortion coefficients are then used in **step 2:** `undistort_image()`, which uses `cv2.undistort()` to correct for the image distortion. An example is shown below:

![text][image1]

### Pipeline (single images)

#### 2. Provide an example of a distortion-corrected image.

Applying all the previous steps to one of the example images, yields the following result:

![text][image2]

That the undistortion was successful can be seen from the difference in the shape of the hood of the car at the bottom of each image.


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In **step 3** I used a combination of color and gradient thresholds to generate a binary image. To this purpose, I explored several combinations of sobel gradient and color thresholds. Each of these was calculated by a separate function, and plotted in its own sub-plot:

* **(b)** `abs_sobel_thresh()`: Calculate gradient orientation in x and y direction
* **(c)** `mag_thresh()`: Calculate gradient magnitude
* **(d)** `dir_thresh()`: Calculate gradient direction
* **(e)** `color_thresh()`: Calculate color threshold

For calculating the color threshold I decided to convert the RGB image to HLS color space applying `cv2.cvtColor(image, cv2.COLOR_RGB2HLS)` and to use only the S-channel to achieve robust identification of the lane lines.   


* **(f)** `combine_thresholded_imgages()`: Combines all thresholds into an output image

The output image will then be used in further processing steps.

![text][image3]

#### 4. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In **step 4** I applied a perspective transform to rectify the binary image from the previous step. In order to calculate the matrix needed for the image rectification, we first have to define source and destination points. The source points have a trapezoid shape along the lane lines, whereas the destination points represent the source points in a rectified image and thus should have a rectangular shape. See below:

![alt text][image4]

I decided to hard-code the points in the following manner:

| Source        | Destination   | Location      |
|:-------------:|:-------------:|:-------------:|
| 310,  665     | 250,  720     | Bottom left   |
| 595, 460      | 250,    0     | Top left      |
| 725,  460     | 1065,   0     | Top right     |
| 1080, 670     | 1065, 720     | Bottom right  |

The points will then be fed in `get_perspective_matrix()` into `cv2.getPerspectiveTransform()`, which computes the perspective transformation matrix. Reversing the order of the points as input parameters returns the inverse of the matrix.

```python
# Calculate the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Calculate the inverse by swapping the input parameters
M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
```  
With the respective transformation matrix the image can either be warped or unwarped by applying `warp_image()`.


#### 5. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In **step 6** we detect the lane lines. This is done in several steps. First, we need to explicitly decide, which pixels are part of the lines and which belong to the left line and which belong to the right line.

As one potential solution, we can plot a histogram with `hist()` indicating where the binary activations occur across the image. We can use the two highest peaks as starting points for determining where the lane lines are located.

![alt text][image5]

Based on the histogram, we can apply sliding windows moving upward in the image (further along the road) to determine where the lane lines go. To this end, we have to define a few hyperparameters related to our sliding windows, and set them up to iterate across the binary activations in the image. Here is an example of my hyperparameters used in the `find_lane_pixels()`:

```python
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
```

The sliding window method goes as follows: We loop through the `nwindows`. During each iteration we find the boundaries of our current window and draw these with `cv2.rectangle*()` onto our visualization image. Note visualization image won't be used in our image processing pipeline. Next, within the boundaries of our window, we check for activated, i.e. nonzero, pixels and append these to our list for left and right line respectively. If the number of activated pixels is greater than the hyperparameter `minpix`, then the window will be re-centered. This procedure is coded by `find_lane_pixels()`.

In the next step, we fit a polynomial with `fit_polynomial()` to the number of pixels belonging to each line. See also visualization image below:

![alt text][image6]

Since we have already detected the lane lines using the sliding window method, we have prior information, which can be used for the lane line detection in the next frame. In this case, we just search for the lines in a margin around the previously fitted polynomials. We get all activated pixels for the left and right line and fit a new set of polynomials. This method is implemented in `find_lines_from_prior()` and an example visualization is shown below. Note that the green shaded area illustrates the new search margin.

![alt text][image7]

#### 6. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in **step 6** in `get_curvature_radius()`. In order to apply radius in the real world we need to convert our pixels values to real world space. This involves measuring how long and wide the section of lane is that we're projecting in our warped image. Let's just assume that the lane is about 30 meters long and 3.7 meters wide. The conversion values are then defined as follows:

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/(right_x_base-left_x_base) # meters per pixel in x dimension
```
*(Note that right_x_base and left_x_base represent the starting positions for the lane lines.)*

We use these values to compute new polynomials in real world space

```python
# Fit new polynomials in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fit_x*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fit_x*xm_per_pix, 2)
```

and calculate the radius for each curve.

```python
# Calculation of R_curve (radius of curvature)
left_curve_rad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curve_rad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```  

Please, refer to this [tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) for more details about the calculations.

---

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code (see also `get_lane_center_offset()`):

```python
# Get car and center lane positions
car_position = image.shape[1]/2
lane_center_position = (left_fit_x + right_fit_x)/2

# Horizontal car offset
lane_center_offset = (car_position - lane_center_position) * xm_per_pix
```

With `xm_per_pix` we again transform the pixel values to real world values. Assuming the camera is mounted at the center of the vehicle, then the car position is calculated as the difference between intercept points and the image midpoint.

#### 7. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In **step 7** we warp the detected lane boundaries back onto the original image. In `draw_lanes()` we draw a polygon between the lane line fits and  warped it back to the perspective of the original image using the inverse perspective matrix `M_inv`. This image is then overlaid onto the original image.

![alt text][image9]

The text identifying the curvature radius and vehicle position has been added by the `draw_data()` function.

---

### Pipeline (video)

#### 8. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video is located in workspace: *./project_video_output.mp4*

---

### Discussion

#### 9. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First of all, I would probably refactor the code. This would include a proper implementation of the Line() class, as well as creating separate classes or modules from functions (e.g. camera calibration).

Second, my code is missing sanity checks - to check for example if the lane lines have similar curvature.

Third, to get better results I would have to implement a more robust search. For example if I lose track of the lines, I would have to revert back to the sliding window search or use other method instead to rediscover the lines again.

Fourth, due to different lightning conditions the static thresholding might cause some problems in the lane line detection. A more dynamic solution or a different method, might yield better results here.

Fifth, In my opinion the pipeline would fail if there were too many cars or several white or yellow cars instead. In the first case no lane lines would be detected and in the latter case the cars might be mistaken for lane lines.
