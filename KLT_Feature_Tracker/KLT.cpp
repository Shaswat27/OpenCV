#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;

Mat img_prev,img_next,gr_prev,gr_next;

/// For FAST
void FAST(Mat img);
int thresh = 30;
std::vector <cv::KeyPoint> keypoints_prev;

int main(int argc, char** argv)
{
	/// Error check
	if (argc!=3)
	{
		printf("Enter previous and next frame!\n");
		return -1;
	}

	/// Load images
	img_prev=imread(argv[1],1);
	img_next=imread(argv[2],1);

	/// Convert to grayscale
	cvtColor( img_prev, gr_prev, CV_BGR2GRAY );
	cvtColor( img_next, gr_next, CV_BGR2GRAY );

	/// Find features in img_prev
	FAST(gr_prev);

	/// Display features
	cv::namedWindow("FAST features",CV_WINDOW_AUTOSIZE);
    Mat corners;
    drawKeypoints( img_prev, keypoints_prev, corners, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    cv::imshow("FAST features", corners);

    
	///  Apply KLT Tracker
	vector<uchar> status;
	vector<float> err;
	std::vector<Point2f> prev,temp,next;
	KeyPoint key;
	key.convert(keypoints_prev,prev);
	std::vector<cv::Mat> pyr;
	int levels;
	levels=buildOpticalFlowPyramid(gr_prev, pyr, Size(21,21), 4);
	calcOpticalFlowPyrLK( pyr, gr_next, prev, temp, status, err);
	size_t i, k;
	Mat draw = imread(argv[1],1);
	
	for( i = k = 0; i < temp.size(); i++ )
    {
    	/// Status = 0 => feature not found
    	if(!status[i]) continue;
    	/// Status = 1 => feature found
    	else
    	{
    		next.push_back(temp[i]);
    		cv::line(draw,prev[i],temp[i],cv::Scalar(255));
    		k++;
    	}
    	    	
    }

    /// Display the image with tracks
    cv::namedWindow("KLT Tracker",CV_WINDOW_AUTOSIZE);
    cv::imshow("KLT Tracker", draw);
    cv::waitKey(0);
}

void FAST(Mat img) 
{
	FASTX(img, keypoints_prev, thresh, true, FastFeatureDetector::TYPE_9_16);
}