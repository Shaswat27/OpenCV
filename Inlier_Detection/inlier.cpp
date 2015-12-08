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
int thresh = 115;
std::vector <cv::KeyPoint> keypoints_prev;

/// For Inlier detection
/// Norm
int match_error = 16;

double norm(Point2f i, Point2f j);

/// Creating intersection of two adjacency lists
std::vector<int> intersection(std::vector<int> v, std::vector<int> temp);

/// Return a vector of potential nodes
std::vector<int> findPotentialNodes(std::vector<int> Q_int, int** W, int k);

/// Update the current clique to obtain maximal clique
std::vector<int> updateClique(std::vector<int> potentialNodes, std::vector<int> Q_int, int** W, int k);

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
            track.push_back(i);
    		cv::line(draw,prev[i],temp[i],cv::Scalar(255));
    		k++;
       	}
    	    	
    }

    /// Inlier detection, there are k elements in next
    /// Construct W
    int** W = (int **)malloc(k * sizeof(int *));
    for (i=0; i<k; i++)
         W[i] = (int *)malloc(k * sizeof(int));
    for(int i=0; i<k; i++)
    {
    	for(int j=0; j<k; j++)
    	{
    		int i_prev=track[i]; /// Find the corresponding feature in previous frame
    		int j_prev=track[j];
            if (i==j) W[i][j]=0;
    		else if ( abs(norm(next[i],next[j])-norm(prev[i_prev],prev[j_prev]))<=match_error ) W[i][j]=1; 
    		else W[i][j]=0;
    	}
    }

    printf("%lu\n",k);

    /*for(int i=0;i<k;i++)
    {
        for(int j=0;j<k;j++)
            printf("%d ",W[i][j]);
        printf("\n");
    }
    */
    
    
    /// Construct maximum clique from W
    std::vector<Point2f> Q;
    /// Initialize clique with node of highest degree
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

    //printf("\nIntial node: %d",node);

    //int o=0;
    
    /// Continue to update clique using greedy algorithm
    while(1)
    {
        std::vector<int> potentialNodes=findPotentialNodes(Q_index,W,k);

        if (potentialNodes.size() == 0) break; /// No more nodes to be found => exit loop
        else
        {
            Q_index = updateClique(potentialNodes,Q_index,W,k);
        }
        /// Keep only unique nodes
        std::vector<int>::iterator it;
        it = std::unique (Q_index.begin(), Q_index.end());
        Q_index.resize( std::distance(Q_index.begin(),it) );
       
    }
    
    /// Final clique is found
    /// Q_index now has indices of all the nodes in the maximal clique
    for(int i=0;i<Q_index.size();i++)
        Q.push_back(next[Q_index[i]]);

    /// Q now has all the mutually consistent features

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

std::vector<int> intersection(std::vector<int> v, std::vector<int> temp) 
{
    std::vector<int> intsctn;

    std::sort(v.begin(), v.end());
    std::sort(temp.begin(), temp.end());
 
    std::set_intersection(v.begin(), v.end(), temp.begin(), temp.end(), std::back_inserter(intsctn));

    return intsctn;
}

std::vector<int> findPotentialNodes(std::vector<int> Q_int, int** W, int k)
{
    /// Vector to store potential nodes
    std::vector<int> potentialNodes, temp, final_set;

    for (int i=0; i<Q_int.size(); i++) /// For all nodes currently in clique
    {
        std::vector<int> v;
        for(int j=0;j<k;j++) /// For nodes connected to the node in clique
        {
            if(W[j][Q_int[i]]==1) v.push_back(j); 
        }
        if (temp.size()==0) temp=final_set=v;
        else 
        {
            final_set = intersection(v,temp);
            temp=final_set;
        }
    }
    return final_set;
}

std::vector<int> updateClique(std::vector<int> potentialNodes, std::vector<int> Q_int, int** W, int k)
{
    int deg=0, max_deg=0, node;

    std::vector<int> Q=Q_int;

    for(int i=0;i<potentialNodes.size();i++)
    {
        deg=0;
        for(int j=0;j<k;j++)
        {
            deg+=W[j][potentialNodes[i]];
        }
        if(deg>=max_deg) 
        {
            max_deg=deg;
            node = potentialNodes[i];
        }       
    }

    Q.push_back(node);
    return Q; 
}
