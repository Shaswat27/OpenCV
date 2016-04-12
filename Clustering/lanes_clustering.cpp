#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv)
{
    const int MAX_CLUSTERS = 5;

    Mat img;
    img = imread(argv[1]);

    imshow("Image", img);

     Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };


    int k, clusterCount = 2;
    int i, sampleCount = 0;
    int j;
    std::vector<Point2f> points;
    Mat labels;

    

    for(i=0; i<img.rows; i++)
    {
    	for(j=0; j<img.cols; j++)
    	{
    		Vec3b BGR = img.at<Vec3b>(i,j);
    		if(BGR.val[0]>230 && BGR.val[1]>230 && BGR.val[2]>230) //detecting white points
    		{
    			Point2f p;
    			p.x = i;
    			p.y = j;
    			points.push_back(p);
    		}
    	}
    }

    sampleCount = points.size();

    clusterCount = MIN(clusterCount, sampleCount);

    printf("Sample count = %d, Cluster count = %d\n", sampleCount, clusterCount);

    Mat inputData(points.size(), 2, CV_32F);
    for(size_t i = 0; i < points.size(); ++i) 
    {
        inputData.at<float>(i, 0) = points[i].y;
        inputData.at<float>(i, 1) = points[i].x;
    }

    Mat clustersCenters;

    static const int cIterations = 10000;
    static const float cEps = 0.001;
    static const int cTrials = 5;

    cv::kmeans(inputData, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, cIterations, cEps), cTrials, cv::KMEANS_PP_CENTERS, clustersCenters);

    printf("%d, %d\n%d, %d\n",clustersCenters.at<uchar>(0,0),clustersCenters.at<uchar>(0,1),clustersCenters.at<uchar>(1,0),clustersCenters.at<uchar>(1,1));

    Mat clusters(img.rows,img.cols, CV_8UC3, Scalar(0,0,0));

    Point center1, center2;
    center1 = clustersCenters.at<Point2f>(0);
    center2 = clustersCenters.at<Point2f>(1);
    
    for( i = 0; i < sampleCount; i++ )
    {
        int clusterIdx = labels.at<int>(i);
        Point ipt = inputData.at<Point2f>(i);;
        circle(clusters, ipt, 2, colorTab[clusterIdx]);//, CV_FILLED, CV_LINE_AA );
    }

    circle(clusters, center1, 20, Scalar(255,255,0));
    circle(clusters, center2, 20, Scalar(0,255,255));

    
    imshow("Clusters", clusters);

    imwrite("cluster.jpg", clusters);
             
    waitKey(0);

    return 0;
}