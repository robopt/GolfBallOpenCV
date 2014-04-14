#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <imgproc/include/opencv2/imgproc/imgproc.hpp>
#include <core/include/opencv2/core/mat.hpp>
#include <math.h>

/* Tracking Golfballs using OpenCV
 * Uses Hough Transform to find golfballs
 * Calculates distance from golfball using an \\
 * equation derivated by Sean Reid using the Lens equation
 *
 * Author: Edward Mead
 */
using namespace cv;

int main()
{

		CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
		if ( !capture ) {
				fprintf( stderr, "ERROR: capture is NULL \n" );
				getchar();
				return -1;
		}
		int rmin = 0, gmin = 0, gmax = 0, bmax = 0;
		int thresh = 50;
		int smooth = 5;

		//UI
		cvNamedWindow("Image:",1);
		cvCreateTrackbar("Thresh", "Image:", &thresh, 255,0);
		cvCreateTrackbar("Smooth", "Image:", &smooth, 100,0);
		cvCreateTrackbar("Rmin", "Image:", &rmin, 255,0);
		cvCreateTrackbar("Gmin", "Image:", &gmin, 255,0);
		cvCreateTrackbar("Gmax", "Image:", &gmax, 255,0);
		cvCreateTrackbar("Bmax", "Image:", &bmax, 255,0);
		cvNamedWindow("Orig:",1);
		CvMemStorage* storage = cvCreateMemStorage(0);

		//video properties
		cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,480);
		cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,320);
		cvSetCaptureProperty(capture,CV_CAP_PROP_FPS,5);
		//attempt to set brightness/contrast using v4lctl
		try{
				system("v4lctl bright 50");
		} catch(...) {}
		try{
				system("v4lctl contrast 5");
		} catch(...) {}
		try{
				system("v4lctl setattr \"Backlight Compensation\" 5");
		} catch(...) {}
		//set brightness/contrast on camera
		//cvSetCaptureProperty(capture,CV_CAP_PROP_BRIGHTNESS,1);
		//cvSetCaptureProperty(capture,CV_CAP_PROP_CONTRAST,.5);
		int filterSize = 3;
		IplConvKernel *convKernel = cvCreateStructuringElementEx(filterSize,filterSize, (filterSize-1)/2,(filterSize-1)/2,CV_SHAPE_RECT,NULL);

		//main loop
		while ( 1 ) {

				// get frame
				IplImage* img = cvQueryFrame( capture );
				IplImage* bimg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);

				//failed to get frame... is camera plugged in?
				if ( !img ) {
						fprintf( stderr, "ERROR: frame is null...\n" );
						getchar();
						break;
				}

				int h = img->height;
				int w = img->width;
				int imgStep = 3; //3 bytes read per interation
				char lastRow[w]; // keep last row in memory/cache
				char lastPixel = 0; //whether the last pixel was a 1/0
				unsigned char *data= reinterpret_cast<unsigned char *> (img->imageData);
				unsigned char *bdata = reinterpret_cast<unsigned char *> (bimg->imageData);
				for (int y = 1; y < h-1; y++)
				{
						for (int x = 1; x < w-1; x++)
						{
								//int* r,g,b;
								//b = (int*)data[((w*y+x)*imgStep)];
								//g = (int*)data[((w*y+x)*imgStep)+1];
								//r = (int*)data[((w*y+x)*imgStep)+2];

								//generate binary image
								if (data[((w*y+x)*imgStep)] <= bmax && data[((w*y+x)*imgStep)+1] >= gmin && data[((w*y+x)*imgStep)+1] <= gmax && data[((w*y+x)*imgStep)+2] >= rmin)
								{
										lastRow[x] = bdata[(w*y+x)] = 255;
								}
								else
								{
										if (data[((w*y+(x-1))*imgStep)] == 255 && lastRow[x] == 255 && (lastRow[x-1] == 255 && lastRow[x+1] == 255) ) //data[((w*y+(x-1))*imgStep)] == 255 && data[((w*y+x)*imgStep)+1] >= gmax && data[((w*y+x)*imgStep)+2] >= gmax &&
												lastRow[x] = bdata[(w*y+x)] = 255;
										else
												lastRow[x] = bdata[(w*y+x)] = 0;
								}
								//if (x == w/2 && y == h/2)
								//printf("R:%d\tG:%d\tB:%d\n",(int)r,(int)g,(int)b);
						}
				}
				cvSmooth(bimg,bimg, CV_MEDIAN, 1, 1,((double)smooth)/10.0,((double)smooth)/10.0);
				cvMorphologyEx(bimg,bimg,NULL,convKernel,CV_MOP_OPEN);

				//noise removal, edge smoothing
				// costs O(n*a) per function
				// where n is # of pixels
				// and a is number of times operation is applied
				//cvErode(bimg,bimg,0,1);
				//cvDilate(bimg,bimg,0,2);
				//cvSmooth(bimg,bimg, CV_GAUSSIAN, 3, 3,1,1);


				//cv Hough Transform to find circles
				CvSeq* circles = cvHoughCircles(bimg, storage, CV_HOUGH_GRADIENT, 2, img->width/10, thresh,thresh/2,0,150);

				//iterate over circles apply reid's equation
				for (size_t i = 0; i < circles->total; i++)
				{
						// round the floats to an int
						float* p = (float*)cvGetSeqElem(circles, i);
						cv::Point center(cvRound(p[0]), cvRound(p[1]));
						int radius = cvRound(p[2]);
						// draw the circle center
						cvCircle(img, center, 3, CV_RGB(0,255,0), -1, 8, 0 );

						// draw the circle outline
						cvCircle(img, center, radius+1, CV_RGB(0,0,255), 2, 8, 0 );

						printf("x: %d y: %d r: %d d: %f\n",center.x,center.y, radius,2*(292*2.1*.74)/(radius-.74));
				}

				//show image
				cvShowImage( "Image:", img);
				cvShowImage( "Orig:", bimg);
				cvReleaseImage(&bimg);

				// Do not release the frame!
				if ( (cvWaitKey(10) & 255) == 27 ) break;
		}

		cvWaitKey();
		cvDestroyWindow("Image:");
		cvDestroyWindow("Orig:");
		cvReleaseCapture( &capture );

		return 0;
}
