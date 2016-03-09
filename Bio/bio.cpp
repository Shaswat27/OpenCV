#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "alglib/ap.h"
#include "alglib/dataanalysis.h"
#include <stdlib.h>
#include <stdio.h>

using namespace alglib;
using namespace cv;

class blob
{
  public:
    float x;
    float y;

    float avg_red;
    float fraction_red;
    float first_red_mode;
    float second_red_mode;
    float difference_modes;
    float count_pixels;
    float absorbance;
    float abs_blue;

    blob()
    {
      x = 0.0;
      y = 0.0;

      avg_red = 0.0;
      fraction_red = 0.0;
      first_red_mode = 0.0;
      second_red_mode = 0.0;
      difference_modes = 0.0;
      count_pixels = 0.0;
      absorbance = 0.0;
      abs_blue = 0.0;
    }

    void set(float x, float y, float avg_red, float fraction_red, float first_red_mode, float second_red_mode, float count_pixels, float absorbance, float abs_blue)
    {
      this->x = x;
      this->y = y;

      this->avg_red = avg_red;
      this->fraction_red = fraction_red;
      this->first_red_mode = first_red_mode;
      this->second_red_mode = second_red_mode;
      this->difference_modes = first_red_mode - second_red_mode;
      this->count_pixels = count_pixels;
      this->absorbance = absorbance;
      this->abs_blue = abs_blue;
    }

    ~blob()
    {}

    void normalize()
    {
      avg_red /= count_pixels;
      first_red_mode /= count_pixels;
      second_red_mode /= count_pixels;
      difference_modes /= count_pixels;
    }

    void view()
    {
      printf("\n%f %f -> %f , %f\n", x, y, absorbance, abs_blue);
    }
};

blob sample[9];

//for modes divide 0-255 as 0-4 5-9 10-14 15-19 ... 250-254 : 255 -> special case
int frequency[51];
std::vector<float> avg_red, fraction_red, first_red_mode, second_red_mode, difference_modes, count_pixels, absorbance, abs_blue, fraction_blue;

void blob_detector(Mat im);

void empty_frequency();
float mode();
void update_freq(int value); 
void calc_features(Mat image, std::vector<KeyPoint> k);

void calc_featuresHSV(Mat im, std::vector<KeyPoint> k);

//machine learning
void _PCA(); //Principle  component analysis

int main( int argc, char** argv )
{
  // Read image
  Mat im;
  im = imread( "crop.jpeg", 1 );

  blob_detector(im);



  //_PCA();

  waitKey(0);
}

void blob_detector(Mat im)
{
  Mat im_blur;

  Size size(200,200);

  resize(im, im, size);

  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params params;
 
  // Change thresholds
  params.minThreshold = 10;
  params.maxThreshold = 255;
 
  // Filter by Area.
  params.filterByArea = true;
  params.minArea = 30;
 
  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.1;
 
  // Filter by Convexity
  //c = conv/10.0;
  params.filterByConvexity = false;
  params.minConvexity = 0.3;
 
  SimpleBlobDetector detector(params);
 
  // Detect blobs.
  std::vector<KeyPoint> keypoints;
  detector.detect( im, keypoints);


  for(int i=0; i<keypoints.size(); i++)
  {
      if(keypoints[i].size < 21) keypoints.erase(keypoints.begin()+i);
  }

  //the keypoints are now stored in keypoints
  /*calc_features(im, keypoints);

  float I0 = fraction_red[8]; //blank
  float I0_b = fraction_blue[8];

  for(int i=0; i<keypoints.size(); i++)
  {
      sample[i].set(keypoints[i].pt.x, keypoints[i].pt.y, avg_red[i], fraction_red[i], first_red_mode[i], second_red_mode[i], count_pixels[i], -log(fraction_red[i]/I0), -log(fraction_blue[i]/I0_b));
  }

  for(int i=0; i<keypoints.size();i++)
  {
    //sample[i].normalize();
    sample[i].view();
  }*/

  calc_featuresHSV(im, keypoints);
 
  // Draw detected blobs as red circles.
  Mat im_with_keypoints;
  drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    // Show blobs
  imshow("keypoints", im_with_keypoints );
}

void calc_features(Mat image, std::vector<KeyPoint> k)
{
  
  for(int i=0; i<k.size(); i++)
  {
    float diameter = k[i].size; //diameter of blob
    float origin_x = k[i].pt.x, origin_y = k[i].pt.y;

    float sum_frac = 0.0, sum = 0.0, count = 0.0, sum_frac_b = 0.0;
    //loop through the square
    for(int j = origin_x-diameter/2.0; j<=origin_x+diameter/2.0; j++)
    {
      for(int l = origin_y-diameter/2.0; l<=origin_y+diameter/2.0; l++)
      {
        if( (l - origin_y)*(l - origin_y) + (j - origin_x)*(j - origin_x) < diameter*diameter/4.0 )
        {
          count = count+1;
          Vec3b BGR = image.at<Vec3b>(l,j);
          
          float temp  = ((float)(BGR.val[2])/sqrt(BGR.val[0]*BGR.val[0]+BGR.val[1]*BGR.val[1]+BGR.val[2]*BGR.val[2]));
          sum_frac = sum_frac + temp;

          temp  = ((float)(BGR.val[1])/sqrt(BGR.val[0]*BGR.val[0]+BGR.val[1]*BGR.val[1]+BGR.val[2]*BGR.val[2]));
          sum_frac_b = sum_frac_b + temp;          

          temp = (float)(BGR.val[2]);
          sum = sum + temp;

          update_freq(BGR.val[2]);
        }
      }
    }

    //find modes 
    float mode1 = mode();
    first_red_mode.push_back(mode1);

    float mode2 = mode(); //goves second mode now
    second_red_mode.push_back(mode2);

    empty_frequency();

    difference_modes.push_back(mode1 - mode2);

    //for each blob
    count_pixels.push_back(count);
    fraction_red.push_back(sum_frac/count);
    fraction_blue.push_back(sum_frac_b/count);
    avg_red.push_back(sum/count);
  }

}

void calc_featuresHSV(Mat im, std::vector<KeyPoint> k)
{

  Mat image;
  cvtColor(im,image, CV_BGR2HSV);

  std::vector<double> H,S,V;


  for(int i=0; i<k.size(); i++)
  {
    float diameter = k[i].size; //diameter of blob
    float origin_x = k[i].pt.x, origin_y = k[i].pt.y;

    float sum_H = 0.0, sum_S = 0.0, count = 0.0, sum_V = 0.0;
    //loop through the square
    for(int j = origin_x-diameter/2.0; j<=origin_x+diameter/2.0; j++)
    {
      for(int l = origin_y-diameter/2.0; l<=origin_y+diameter/2.0; l++)
      {
        if( (l - origin_y)*(l - origin_y) + (j - origin_x)*(j - origin_x) < diameter*diameter/4.0 )
        {
          count = count+1;
          Vec3b HSV = image.at<Vec3b>(l,j);
          
          sum_H += HSV.val[0];
          sum_S += HSV.val[1];
          sum_V += HSV.val[2];
        }
      }
    }

    H.push_back(sum_H/count);
    S.push_back(sum_S/count);
    V.push_back(sum_V/count);
  }


  for(int i=0; i<k.size();i++)
    printf("\n%f %f -> %lf , %lf, %lf\n", k[i].pt.x, k[i].pt.y, H[i], S[i], V[i]);
}

void update_freq(int value)
{
  int index = value/5;

  if(index < 51)
    (frequency[index])++;
  else if (index == 51)
    (frequency[50])++;
}

float mode()
{
  int largest = 0;

  for(int i=0; i<51; i++)
  {
    if( frequency[i] > frequency[largest] )
      largest = i;
  }

  float m = (float)((largest*5 + largest*5 + 4))/2.0; //average of the 5 sized bin 

  frequency[largest] = 0; //to facilitate finding the second mode

  return m;
}

void empty_frequency()
{
  for(int i=0; i<51; i++)
    frequency[i] = 0;
}

void _PCA()
{
  double descriptors[54];

  for(int i=0, k=0; i<9; i++)
  {
    //sample[i].normalize();
    for(int j=0; j<6; j++)
    {
      switch(j)
      {
        case 0:
                descriptors[k++] = sample[i].avg_red;
                break;
        case 1:
                descriptors[k++] = sample[i].fraction_red;
                break;
        case 2:
                descriptors[k++] = sample[i].first_red_mode;
                break;
        case 3:
                descriptors[k++] = sample[i].second_red_mode;
                break;
        case 4:
                descriptors[k++] = sample[i].difference_modes;
                break;
        case 5:
                descriptors[k++] = sample[i].count_pixels;
                break;
      }
    }
  }

  // add this data to ALGLIB's format
  alglib::real_2d_array ptInput;
  ptInput.setcontent(9, 6, descriptors);

  // this guy gets passed in and is filled in with an integer status code
  alglib::ae_int_t info;

  // scalar values that describe variances along each eigenvector
  alglib::real_1d_array eigValues;

  // unit eigenvectors which are the orthogonal basis that we want
  alglib::real_2d_array eigVectors;

  // perform the analysis
  pcabuildbasis(ptInput, 9, 6, info, eigValues, eigVectors);

  // now the vectors can be accessed as follows:
  double basis[6][9]; //arranged column-wise

  for(int i=0; i<6; i++)
  {
    for(int j=0; j<9; j++)
    {
      basis[i][j] = eigVectors[i][j];
      printf("%lf\t", basis[i][j]);
    }  
    printf("\n");
  }

  /*
  double basis0_0 = eigVectors[0][0];
  double basis0_1 = eigVectors[1][0];
  double basis0_2 = eigVectors[2][0];
  double basis0_3 = eigVectors[3][0];
  double basis0_4 = eigVectors[4][0];
  double basis0_5 = eigVectors[5][0];*/
}
