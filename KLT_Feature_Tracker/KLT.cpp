#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

Mat img_prev,img_next,gr_prev,gr_next;

/// For FAST
std::vector<cv::KeyPoint> FAST(Mat img);
int thresh = 60;
std::vector <cv::KeyPoint> keypoints_prev;
std::vector <cv::KeyPoint> keypoints_next;



int main(int argc, char** argv)
{
	/// Error check
	if (argc!=3)
	{
		printf("Enter previous and next frame!\n");
		return -1;
	}

	/// Load images
	img_prev=imread(argv[1]);
	img_next=imread(argv[2]);

	/// Convert to grayscale
	cvtColor( img_prev, gr_prev, CV_BGR2GRAY );
	cvtColor( img_next, gr_next, CV_BGR2GRAY );

	/// Find features in img_prev
	keypoints_prev=FAST(gr_prev);

	///  Apply KLT Tracker
	vector<uchar> status;
	vector<float> err;
	std::vector<Point2f> prev,next;
	prev = keypoints_prev.pt;
	calcOpticalFlowPyrLK( gr_prev, gr_next, prev, next, status, err);
	size_t i, k;
    for( i = k = 0; i < next.size(); i++ )
    {
    	
    }
}

std::vector<cv::KeyPoint> FAST(Mat img)
{
	std::vector<cv::KeyPoint> keypoints;
	FASTX(img, keypoints, thresh, true, FastFeatureDetector::TYPE_9_16);
	return keypoints;
}