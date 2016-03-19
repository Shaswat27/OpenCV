#include "../include/ht.hpp"

int main(int argc, char *argv[])
{
	Mat frame, back, fore;

	srand(time(NULL));

	VideoCapture cap(0);
	cap >> frame;
	Mat black(frame.rows, frame.cols, CV_8UC3, Scalar(0,0,0));
	BackgroundSubtractorMOG2 bg; bg.set("nmixtures",3); bg.set("detectShadows",false);

	int backgroundFrame = 500; //to calculate average background image

	while(1)
	{
		//Get the frame
		cap >> frame;

		//Update the current background model and get the foreground
		if(backgroundFrame>0)
		{
			bg.operator ()(frame,fore);
			backgroundFrame--;
		}
		else
		{
			bg.operator()(frame,fore,0);
		}

		//enhance edges in the foreground
		erode(fore,fore,Mat());
		dilate(fore,fore,Mat());

		vector<vector<Point> > hulls;
		
		hulls = find_contours(&frame, fore);

		if(hulls.size()>=2 && hulls.size()<=3)
			remove_color(&black, frame, hulls);

		imshow("Frame",frame);
		imshow("Black", black);

		if(waitKey(10)>=0)break;		
				
	}
	return 0;
}