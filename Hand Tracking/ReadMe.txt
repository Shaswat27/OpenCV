Approach to the problem:
Step 1: Create a VideoCapture object to capture frames from the laptop's camera
Step 2: Detection of foreground (background segementation using mixture of gaussians approach)
		-> Calculate the average background image
		-> When each frame is captured, apply background segmentation. The foreground (difference of the current image with respect to the 
		   average background image) would contain the hands (moving).
Step 3: Apply morphological operations on the segmented foreground to enhance the edges
Step 4: Detect contours in the foreground
Setp 5: If the contour is significant (i.e. rejecting contours with smaller areas) then find the convex hull encompassing that contour
Step 6: If the number of convex hulls detected is >=2 && <=3 then most likely the two hands are detected in the foreground
Step 7: In that case, mark the corresponding areas as white in the black dummy image thereby recreating the cleaning pattern

Instructions:
Running the build: Double click on "handTracking" or navigate to the folder in terminal and run "./handTracking"
Creating a new build (after changes): Only for the first time - Navigate to the folder in terminal and run "cmake ."
									  Run the command "make"
									  The new build is created

Possible Bugs:
-> If there is too much movement in the scene additional contours (other than the hands) will also be detected
-> The threshold for the contour to be significant is set at area>=5000 (in find_contours (include/ht.hpp)). This might vary with difference in 
   resolution of the frames captured by different laptop cameras. [RECTIFICATION: edit the value and create a new build]

NOTES:
-> For marking the areas white, tried three approaches:
	1. Simply iterate through the image and mark those areas white which inside the hulls
	2. Calculate the centroid of each hull and apply BFS/DFS to mark the areas as white [since the hull is a convex polygon, the centroid always
	   lies inside it]
	3. Use openCV's function for drawing filled contours (which is a highly optimised version of simple DFS)
   The first two approaches were very slow for the frame rate and hence settled with the third approach.
-> It would be cool to have an adroid implementation of this.
	Sample OpenCV code (C++) on Android (using android-ndk for creating a Java wrapper of C++ code) - 
	https://github.com/Shaswat27/android/tree/master/TestOpenCV [converts incoming frames to grayscale, detects corners in it using FAST and 
	displays the result]
   This can be easily extended to work in any other OpenCV project since the wrapper is already made. 


 
