#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

//find countours and convex hull
vector<vector<Point> > find_contours(Mat *frame, Mat fore)
{
	vector<vector<Point> > contours;
	vector<vector<Point> > hull;

	//find the contours in the foreground
	findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_TC89_L1);
	for(int i=0;i<contours.size();i++)
	{
		//reject the ones with insignifaccnt areas
		if(contourArea(contours[i])>=5000)		    
		{
			vector<vector<Point> > tcontours;
			tcontours.push_back(contours[i]);;

			//detect the convex hull
			vector<vector<Point> > hulls(1);
			vector<vector<int> > hullsI(1);
			convexHull(Mat(tcontours[0]),hulls[0],false);
			convexHull(Mat(tcontours[0]),hullsI[0],false);

			hull.push_back(hulls[0]);

			//draw the contour
			drawContours(*frame,hulls,-1,cv::Scalar(0,255,0),2);
		}
	}

	return hull;
}

//remove color from the image corresponding to the hulls
void remove_color(Mat *black, Mat frame, vector<vector<Point> > hulls)
{
	for(int k = 0; k<hulls.size(); ++k)
	{
		double x = 0.0, y = 0.0;

		for(int h = 0; h<hulls[k].size(); ++h)
		{
			x += hulls[k][h].x;
			y += hulls[k][h].y;
		}

		x /= hulls[k].size();
		y /= hulls[k].size();

		Point seed;
		seed.x = x;
		seed.y = y;

		vector<vector<Point> > _hulls(1);

		_hulls.push_back(hulls[k]);

		drawContours(*black,_hulls,-1,cv::Scalar(255,255,255),CV_FILLED);
    }
}

