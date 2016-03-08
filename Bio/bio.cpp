#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

class blob
{
  private:
    float x;
    float y;

    float avg_red;
    float fraction_red;
    float first_red_mode;
    float second_red_mode;
    float difference_modes;
    float count_pixels;

  public:
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
    }

    void set(float x, float y, float avg_red, float fraction_red, float first_red_mode, float second_red_mode, float count_pixels)
    {
      this->x = x;
      this->y = y;

      this->avg_red = avg_red;
      this->fraction_red = fraction_red;
      this->first_red_mode = first_red_mode;
      this->second_red_mode = second_red_mode;
      this->difference_modes = first_red_mode - second_red_mode;
      this->count_pixels = count_pixels;
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
      printf("\n%f %f -> %f , %f, %f, %f, %f\n", x, y, avg_red, first_red_mode, second_red_mode, fraction_red, difference_modes);
    }
};

blob sample[9];

//for modes divide 0-255 as 0-4 5-9 10-14 15-19 ... 250-254 : 255 -> special case
int frequency[51];
std::vector<float> avg_red, fraction_red, first_red_mode, second_red_mode, difference_modes, count_pixels;

void blob_detector(Mat im);

void empty_frequency();
float mode();
void update_freq(int value); 
void calc_features(Mat image, std::vector<KeyPoint> k);

int main( int argc, char** argv )
{
  // Read image
  Mat im;
  im = imread( "crop.jpeg", 1 );

  blob_detector(im);

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
  calc_features(im, keypoints);

  for(int i=0; i<keypoints.size(); i++)
  {
      sample[i].set(keypoints[i].pt.x, keypoints[i].pt.y, avg_red[i], fraction_red[i], first_red_mode[i], second_red_mode[i], count_pixels[i]);
  }

  for(int i=0; i<keypoints.size();i++)
  {
    sample[i].normalize();
    sample[i].view();
  }
 
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

    float sum_frac = 0.0, sum = 0.0, count = 0.0;
    //loop through the square
    for(int j = origin_x-diameter/2.0; j<=origin_x+diameter/2.0; j++)
    {
      for(int l = origin_y-diameter/2.0; l<=origin_y+diameter/2.0; l++)
      {
        if( (l - origin_y)*(l - origin_y) + (j - origin_x)*(j - origin_x) < diameter*diameter/4.0 )
        {
          count = count+1;
          Vec3b BGR = image.at<Vec3b>(l,j);
          
          float temp  = ((float)(BGR.val[2])/(BGR.val[0]+BGR.val[1]+BGR.val[2]));
          sum_frac = sum_frac + temp;

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
    avg_red.push_back(sum/count);
  }

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
