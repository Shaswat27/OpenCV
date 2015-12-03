#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src, src_gray, corners;
int thresh = 12;
int max_thresh = 255;
std::vector <cv::KeyPoint> keypoints;

char* source_window = "Source image";
char* corners_window = "Corners detected";

void FAST_demo(int,void*);

int main( int argc, char** argv )
{
  /// Argument check
  if( argc != 2 ) 
    { 
      printf("Specify image to be opened \n"); 
      return -1; 
    }

  /// Load source image and convert it to gray
  src = imread( argv[1], 1 );
  
  /// Error check
  if( !src.data ) 
    { 
      printf("Error loading image \n"); 
      return -1; 
    }

  /// Convert to grayscale for corner detection
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window and a trackbar
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, FAST_demo);
  imshow( source_window, src );

  FAST_demo(0,0);
  
  waitKey(0);
  return(0);
}

void FAST_demo(int,void*)
{

  /// Detecting corners
  FASTX(src_gray, keypoints, thresh, true, FastFeatureDetector::TYPE_9_16);
 
  /// Display corners
  drawKeypoints( src, keypoints, corners, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow("corners_window", corners );   
}