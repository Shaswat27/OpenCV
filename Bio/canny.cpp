#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

int circ = 0, conv = 0, min_t = 10, max_t = 200, min_ar = 1500, min_in = 0;
float max_circ = 10, conv_max = 10, t_max = 200, ar_max = 100000, in_max = 100;

Mat im;

void blob_detector(int, void*);
void canny_circle();

int main( int argc, char** argv )
{
  // Read image
  im = imread( "samples.png", 1 );
 
  /*namedWindow( "BIO", CV_WINDOW_AUTOSIZE );

  createTrackbar( "Minimum Threshold: ", "BIO", &min_t, t_max, blob_detector);
  createTrackbar( "Maximum Threshold: ", "BIO", &max_t, t_max, blob_detector);
  createTrackbar( "Area: ", "BIO", &min_ar, ar_max, blob_detector);
  createTrackbar( "Circularity: ", "BIO", &circ, max_circ, blob_detector); 
  createTrackbar( "Convexity: ", "BIO", &conv, conv_max, blob_detector);
  createTrackbar( "Inertia: ", "BIO", &min_in, in_max, blob_detector);*/

  canny_circle();

  waitKey(0);
}

void blob_detector(int, void*)
{
  Mat im_blur;

  GaussianBlur( im, im_blur, Size(5, 5), 2, 2 );

  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params params;
 
  // Change thresholds
  params.minThreshold = min_t;
  params.maxThreshold = max_t;
 
  // Filter by Area.
  //params.filterByArea = true;
  //params.minArea = min_ar;
 
  // Filter by Circularity
  float c = circ/10.0;
  params.filterByCircularity = true;
  params.minCircularity = c;
 
  // Filter by Convexity
  c = conv/10.0;
  params.filterByConvexity = true;
  params.minConvexity = c;
 
  // Filter by Inertia
  //c = min_in/100.0;
  //params.filterByInertia = true;
  //params.minInertiaRatio = c;

  SimpleBlobDetector detector(params);
 
  // Detect blobs.
  std::vector<KeyPoint> keypoints;
  detector.detect( im, keypoints);
 
  // Draw detected blobs as red circles.
  // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
  Mat im_with_keypoints;
  drawKeypoints( im_blur, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    // Show blobs
  imshow("keypoints", im_with_keypoints );
}

void canny_circle()
{
  Mat im_blur;
  cvtColor(im, im_blur,CV_BGR2GRAY);
  GaussianBlur( im_blur, im_blur, Size(3, 3), 2, 2 );

  vector<Vec3f> circles;

  /// Apply the Hough Transform to find the circles
  HoughCircles( im_blur, circles, CV_HOUGH_GRADIENT, 1, im_blur.rows/8, 200, 100, 0, 0 );

  /// Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( im, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( im, center, radius, Scalar(0,0,255), 3, 8, 0 );
   }

  /// Show your results
  namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
  imshow( "Hough Circle Transform Demo", im );
}