#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findPupils.h"


/** Global variables */
static cv::RNG rng(12345);
static cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);


/**
 * @function load camera parameters
 * added by zhantao deng @EPFL 30/01/2019
 */
void read_param(cv::Mat & camera_Mat, cv::Mat & distCoeffs, cv::Size imageSize, const char * IntrinsicsPath)
{
       bool FSflag = false;
       cv::FileStorage readfs;

       FSflag = readfs.open(IntrinsicsPath, cv::FileStorage::READ);
       if (FSflag == false){
           std::cout << "Cannot open camera_param file" << std::endl;
           exit(0);
       }
       readfs["camera_matrix"] >> camera_Mat;
       readfs["distortion_coefficients"] >> distCoeffs;
       readfs["image_width"] >> imageSize.width;
       readfs["image_height"] >> imageSize.height;

       std::cout << camera_Mat << std::endl << distCoeffs << std::endl << imageSize << std::endl;

       readfs.release();
}

/**
 * @function main
 */
int main( int argc, const char** argv ) {

  PupilsTracker Tracker;

  cv::Mat frame, undist_frame;
  int64 t0 = cv::getTickCount();
  size_t nFrames = 1;

  Tracker.Initializer();
  // create eye-coner kernel
  Tracker.createCornerKernels();

  // draw an ellipse on the given image at given coordinates
  /* cv::Size(23.4, 15.2): Half of the size of the ellipse main axes.
     43.0                  Ellipse rotation angle in degrees.
     0.0                   Starting angle of the elliptic arc in degrees.
     360.0                 Ending angle of the elliptic arc in degrees.
     White                 color Ellipse color.
     -1                    Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that a filled ellipse sector is to be drawn.  */
  ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

  cv::Size imageSize;
  cv::Mat camera_Mat, distCoeffs;
  const char * paramPath = "/home/zhantao/Documents/BasicCVProgram/laptopcamera.yml";
  read_param(camera_Mat, distCoeffs, imageSize, paramPath);

  // Original auther makes an attempt at supporting both 2.x and 3.x OpenCV
#if CV_MAJOR_VERSION < 3
  CvCapture* capture = cvCaptureFromCAM( 0 );
  if( capture ) {
    while( true ) {
      frame = cvQueryFrame( capture );
#else
  cv::VideoCapture capture(0);
  if( capture.isOpened() ) {
    while( true ) {
      capture.read(frame);
#endif


      undistort(frame, undist_frame, camera_Mat, distCoeffs);
      undist_frame.copyTo(frame);

      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(Tracker.debugImage);

      // Apply the classifier to the frame
      if( !frame.empty() ) {
        Tracker.DetectPupils( frame );
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      imshow(Tracker.main_window_name,Tracker.debugImage);

      // count FPS and time per frame
      if (nFrames%10 == 0)
      {
          int64 t1 = cv::getTickCount();
          std::cout << "Average time per frame: " << cv::format("%9.2f ms", (double)(t1 - t0) * 1000.0f / (10 * cv::getTickFrequency())) << std::endl;
          t0 = t1;
      }
      nFrames ++;

      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }

    }
  }

  Tracker.releaseCornerKernels();

  return 0;
}
