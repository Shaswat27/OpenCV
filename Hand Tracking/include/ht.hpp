#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

#define PI 3.14159265359

using namespace std;
using namespace cv;

//square of the distance between 2 points
double dist(Point x,Point y)
{
	return (x.x-y.x)*(x.x-y.x)+(x.y-y.y)*(x.y-y.y);
}

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
			drawContours(*frame,hulls,-1,cv::Scalar(0,0,0),2);
		}
	}

	return hull;
}

double subtend_angle(Point a, Point b, Point test) //return angle subtended by the three points, considering a->b counterclock as +ve angle
{
	Point vect_a, vect_b; //represent vectors from test->a and test->b

	vect_a.x = a.x - test.x; vect_a.y = a.y - test.y;
	vect_b.x = b.x - test.x; vect_b.y = b.y - test.y;

	double mag_a = sqrt( vect_a.x*vect_a.x + vect_a.y*vect_a.y );
	double mag_b = sqrt( vect_b.x*vect_b.x + vect_b.y*vect_b.y );

	double dot_product = vect_a.x*vect_b.x + vect_a.y*vect_b.y;

	double angle_rad = atan2(vect_b.y, vect_b.x) - atan2(vect_a.y, vect_a.x); //angle from a->b, directed

	while (angle_rad > PI)
      angle_rad -= 2*PI;
   	while (angle_rad < -PI)
      angle_rad += 2*PI;

	return angle_rad;	
}

bool check_inside(vector<Point> poly, Point test) //the points would be in clockwise or anti-clockwise order
{
	double angle_subtended = 0.0;

	for(int i=0; i<poly.size()-1; i++)
	{
		angle_subtended += subtend_angle(poly[i], poly[i+1], test);
	}

	if(angle_subtended<0.1 || angle_subtended>-0.1) //allow for some error
		return false;
	else
		return true;
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

		//Scalar replace(255, 255, 255);

		//floodFill(*black, seed, replace);
    }
}

