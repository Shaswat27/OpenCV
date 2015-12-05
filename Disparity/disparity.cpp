#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Left and right images must be undistorted and rectified

int main(int argc, char** argv)
{
	/// Expects ./disparity left_image_path right_image_path
	if (argc!=3)
	{
		printf("Enter adress of left and right images!");
		return -1;
	}

	Mat img_l,img_r,gr_l,gr_r,disp,disp8;
	
	/// Read images
	img_l=imread(argv[1]);
	img_r=imread(argv[2]);

	/// Convert to grayscale
    cvtColor( img_l, gr_l, CV_BGR2GRAY );
    cvtColor( img_r, gr_r, CV_BGR2GRAY );

    /// Create Semi-Global Block Matcher object
    StereoSGBM sgbm;
   
    sgbm.SADWindowSize = 9;
	sgbm.numberOfDisparities = 192;
	sgbm.preFilterCap = 4;
	sgbm.minDisparity = -64;
	sgbm.uniquenessRatio = 1;
	sgbm.speckleWindowSize = 150;
	sgbm.speckleRange = 2;
	sgbm.disp12MaxDiff = 10;
	sgbm.fullDP = false;
	sgbm.P1 = 600;
	sgbm.P2 = 2400;

    /// Create disparity map
    sgbm(gr_l,gr_r,disp);
    

    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

    /// Display the images
    imshow("Left",img_l);
    imshow("Right",img_r);
    imshow("Disparity8",disp8);
    
    /// Wait for user
    waitKey(0);

    return(0);
}