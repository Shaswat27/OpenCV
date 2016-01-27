#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using namespace cv;
using namespace std;
using namespace Eigen;

int frame_rate = 0;

static const std::string OPENCV_WINDOW = "Image window";

Mat img_prev,img_next,gr_prev,gr_next;

Mat disp_prev, disp_next;

int size_Q;

/// For FAST
void FAST(Mat img);
int thresh = 130;
std::vector <cv::KeyPoint> keypoints_prev;

/// For Inlier detection
/// Norm
int match_error = 5;

double norm(Point2f i, Point2f j);

/// Creating intersection of two adjacency lists
std::vector<int> intersection(std::vector<int> v, std::vector<int> temp);

/// Return a vector of potential nodes
std::vector<int> findPotentialNodes(std::vector<int> Q_int, int** W, int k);

/// Update the current clique to obtain maximal clique
std::vector<int> updateClique(std::vector<int> potentialNodes, std::vector<int> Q_int, int** W, int k);

/// main function to compute optical flow
int compute_inliers(Mat img_prev, Mat img_next);

bool flag=false;

// datatype for storing coordinates
typedef struct c
{
  double x;
  double y;
  double d;
  double one;
} coordinates;

// vector to store pixel coordinates
std::vector<coordinates> p_prev, p_next;
// vector to store world coordinates
std::vector<coordinates> w_prev, w_next;

//function to convert pixel to world coordinates
std::vector<coordinates> p_to_w(std::vector<coordinates> p);

//define camera parameters
double Tx = -0.070103;
double cx = 517.5;
double cy = 277.5;
double f =  570.472;

MatrixXd P(3,4);

//routines for minization using Levenberg-Macquardt
Eigen::VectorXd x(12); //vector of parameters

MatrixXd T(4,4);
// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
typedef _Scalar Scalar;
enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
};
typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

int m_inputs, m_values;

Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

int inputs() const { return m_inputs; }
int values() const { return m_values; }

};

struct my_functor : Functor<double>
{
  my_functor(void): Functor<double>(1,1)
  {
    P(0,0) = f;   P(0,1) = 0.0; P(0,2) = cx;  P(0,3) = 0.0;
    P(1,0) = 0.0; P(1,1) = f;   P(1,2) = cy;  P(1,3) = 0.0;
    P(2,0) = 0.0; P(2,1) = 0.0; P(2,2) = 1.0; P(2,3) = 0.0;

    T(0,0) = x(0); T(0,1) = x(1); T(0,2) = x(2); T(0,3) = x(9);
    T(1,0) = x(3); T(1,1) = x(4); T(1,2) = x(5); T(1,3) = x(10);
    T(2,0) = x(6); T(2,1) = x(7); T(2,2) = x(8); T(2,3) = x(11);
    T(3,0) = 0.0;  T(3,1) = 0.0;  T(3,2) = 0.0;  T(3,3) = 1.0;
  }
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
  {
    //implement y = sum of reprojection errors
    MatrixXd temp(1,1);
    temp(0,0) = 0.0;
    for(int i=0; i<size_Q; i++)
    {
      Vector3d j_t(p_prev[i].x, p_prev[i].y, p_prev[i].d);
      Vector3d j_t1(p_next[i].x, p_next[i].y, p_next[i].d);

      Vector4d w_t(w_prev[i].x, w_prev[i].y, w_prev[i].d, w_prev[i].one);
      Vector4d w_t1(w_next[i].x, w_next[i].y, w_next[i].d, w_next[i].one);

      Vector3d a(0.0,0.0,0.0);
      a = j_t - P*T*w_t1;

      Vector3d b(0.0,0.0,0.0);
      b = j_t1 - P*(T.inverse())*w_t;

      temp = temp + a*a.transpose() + b*b.transpose();
    }
    fvec(0) = temp(0,0);
    return 0;
  }
};

//function to minimize reprojection error
void minimize();

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber disp_sub_;
    
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed 
      image_sub_ = it_.subscribe("/multisense/left/image_rect", 30, 
      &ImageConverter::imageCb, this);
    // Subscribe to input disparity feed
      disp_sub_ = it_.subscribe("/multisense/left/disparity", 30, 
      &ImageConverter::dispCb, this); 
    //image_pub_ = it_.advertise("/image_converter/output_video", 1);

    //cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void dispCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr disp_ptr;
    try
    {
      disp_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    if(!flag) //first call
    {
      disp_prev = disp_ptr->image;
      disp_next = disp_ptr->image;
    }

    else
    {
      if(frame_rate%2==0)
        disp_next = disp_ptr->image;
    }
  
    disp_prev = disp_next;   
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    if(!flag) //first call
    {
      img_prev = cv_ptr->image;
      img_next = cv_ptr->image;
      flag = true;
    }

    else
    {
      if(frame_rate>100) frame_rate=0;
      if(frame_rate%2==0)
      img_next = cv_ptr->image;
      frame_rate++;
      size_Q = compute_inliers(img_prev, img_next);
      //convert features to world coordinates
      w_prev = p_to_w(p_prev);
      w_next = p_to_w(p_next);
      std::cout<<"\nImage world coordinates: "<<(p_next.front()).x<<","<<(p_next.front()).y<<","<<(p_next.front()).d<<","<<(p_next.front()).one<<"\n";
      std::cout<<"\nReal world coordinates: "<<(w_next.front()).x<<","<<(w_next.front()).y<<","<<(w_next.front()).d<<","<<(w_next.front()).one<<"\n";
      //minimize();
    }
  
    img_prev = img_next;  
    p_prev.erase(p_prev.begin(),p_prev.begin()+p_prev.size());
    p_next.erase (p_next.begin(),p_next.begin()+p_next.size());
    w_prev.erase (w_prev.begin(),w_prev.begin()+w_prev.size());
    w_next.erase (w_next.begin(),w_next.begin()+w_next.size());
  }
  
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
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

int compute_inliers(Mat img_prev, Mat img_next)
{
    /// Find features in img_prev
  FAST(img_prev);

  /// Display features
  //cv::namedWindow("FAST features",CV_WINDOW_AUTOSIZE);
  //Mat corners;
  //drawKeypoints(img_prev, keypoints_prev, corners, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  //cv::imshow("FAST features", corners);

    
  ///  Apply KLT Tracker
  vector<uchar> status;
  vector<float> err;
  std::vector<Point2f> prev,temp,next;
  std::vector<int> track; // To match pair of feautures
  KeyPoint key;
  key.convert(keypoints_prev,prev);
  std::vector<cv::Mat> pyr;
  int levels;
  levels=buildOpticalFlowPyramid(img_prev, pyr, Size(21,21), 4);
  calcOpticalFlowPyrLK( pyr, img_next, prev, temp, status, err);
  size_t i, k;
  //Mat draw = img_next;
    
  for( i = k = 0; i < temp.size(); i++ )
    {
      /// Status = 0 => feature not found
      if(!status[i]) continue;
      
      /// Status = 1 => feature found 
      else
      {
        next.push_back(temp[i]);
            track.push_back(i);
        //cv::line(draw,prev[i],temp[i],cv::Scalar(255));
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

    //printf("%lu\n",k);

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
    {
      Q.push_back(next[Q_index[i]]);
      
      coordinates p,n;

      n.x = next[Q_index[i]].x;
      n.y = next[Q_index[i]].y;
      n.d = disp_next.at<uchar>(Point(n.x,n.y));
      n.one = 1.0;
      p_next.push_back(n);

      p.x = prev[track[Q_index[i]]].x;
      p.y = prev[track[Q_index[i]]].y;
      p.d = disp_prev.at<uchar>(Point(p.x,p.y));
      p.one = 1.0;
      p_prev.push_back(p);
    }    

    /// Q now has all the mutually consistent features
    //printf("%lu\n", Q.size());

    Mat inlier  = img_next;

    for( int i=0; i < Q_index.size(); i++ )
    {
        cv::line(inlier,prev[track[Q_index[i]]],Q[i],cv::Scalar(255)); 
    }
      
    /// Display the image with tracks
    //cv::namedWindow("KLT Tracker",CV_WINDOW_AUTOSIZE);
    //cv::imshow("KLT Tracker", draw);

    cv::namedWindow("Inlier",CV_WINDOW_AUTOSIZE);
    cv::imshow("Inlier", inlier);

    cv::waitKey(1);

    return Q.size();
}

std::vector<coordinates> p_to_w(std::vector<coordinates> p)
{
  std::vector<coordinates> w;
  for(int i=0;i<p.size();i++)
  {
    coordinates temp;
    temp.one = -p[i].d/Tx;

    temp.x = (p[i].x-cx)/temp.one;
    temp.y = (p[i].y-cy)/temp.one;
    temp.d = -f/temp.one;
    temp.one = 1.0;

    w.push_back(temp);
  }

  return w;

}

void minimize()
{
  for(int i=0;i<12;i++)
  {
    x(i) = i;
  }

  my_functor functor;
  Eigen::NumericalDiff<my_functor> numDiff(functor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>,double> lm(numDiff);
  lm.parameters.maxfev = 2000;
  lm.parameters.xtol = 1.0e-10;
  //std::cout << lm.parameters.maxfev << std::endl;

  int ret = lm.minimize(x);
  //std::cout << lm.iter << std::endl;
  //std::cout << ret << std::endl;

  //std::cout << "x that minimizes the function: " << x << std::endl;

}