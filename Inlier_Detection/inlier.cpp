#include <stdio.h>
#include <iostream>
#include <math.h>
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

/// Norm
double norm(Point2f i, Point2f j);

/// Checking if vertex exists in list
int notBelong(std::vector<int> v, int j);

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
	std::vector<int> track; // To match pair of feautures
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
            track[k]=i;
    		cv::line(draw,prev[i],temp[i],cv::Scalar(255));
    		k++;
       	}
    	    	
    }

    /// Inlier detection, there are k elements in next
    /// Construct W
    int** W = (int **)malloc(k * sizeof(int *));
    for (i=0; i<k; i++)
         W[i] = (int *)malloc(k * sizeof(int));
    /// Fill in values
    std::vector<Point2f> inlier;
    for(int i=0; i<k; i++)
    {
    	for(int j=0; j<k; j++)
    	{
    		int i_prev=track[i]; /// Find the corresponding feature in previous frame
    		int j_prev=track[j];
    		if ( norm(next[i],next[j]) == norm(prev[i_prev],prev[j_prev]) ) W[i][j]=1; 
    		else W[i][j]=0;
    	}
    }
    /// Construct maximum clique from W
    std::vector<Point2f> Q;
    std::vector<int> Q_index;
    int sum=0,tmp=0,node;
    /// Initialize clique with node of highest degree
    for(int i=0;i<k;i++)
    {
        tmp=0; 
        for(int j=0;j<k;j++)
        {
            tmp+=W[i][j];
        }
        if(tmp>=sum)
        {
            sum=tmp;
            node=i;
        }
    }
    Q_index.push_back(node);
    while(1)
    {
        std::vector<int> v;
        for(int i=0;i<Q_index.size();i++) /// For each vertex in Q
        {
            for(int j=0;j<k;j++) /// Check adjacency list of that vertex
            {
                if(W[j][i]==1 && notBelong(v,j)) v.push_back(j);
            }
            /// v[i] now has unique vertices connected to the element in Q
        }
        //Now, select the vertex in 

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

double norm(Point2f i, Point2f j)
{
	return ( sqrt( (i.x-j.x)*(i.x-j.x) + (i.y-j.y)*(i.y-j.y) ) );
}

int notBelong(std::vector<int> v, int j) /// Return 1 if it does not belong to v, else 0
{
    int ret=1;
    for(int i=0;i<v.size();v++)
        if(v[i]==j) ret=0;
    return ret;
}