#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findPupils.h"
#include "optEyePose.h"


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

    size_t nFrames = 1;
    cv::Mat frame, undist_frame;
    int64 t0 = cv::getTickCount();
    cv::Size imageSize;
    cv::Mat camera_Mat, distCoeffs;
    const char * paramPath = "/home/zhantao/Documents/BasicCVProgram/laptopcamera.yml";

    std::vector<cv::Point> vLeftEyePosition, vRightEyePosition;
    std::vector<cv::Point> vOptLeftEyePosition, vOptRightEyePosition;

    PupilsTracker Tracker;
    Tracker.Initializer();
    Tracker.createCornerKernels();

    read_param(camera_Mat, distCoeffs, imageSize, paramPath);
    SGDOptimizer Optimizer( 0.5*(camera_Mat.at<double>(0,0) + camera_Mat.at<double>(1,1)) );


// the Original author makes an attempt at supporting both 2.x and 3.x OpenCV
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

                Optimizer.receiver(Tracker.left_eyepos, Tracker.right_eyepos);
                if (Optimizer.initiated)
                {
                    circle(Tracker.debugImage, cv::Point( Optimizer.voptleftx(iInitialFrames - 1),  Optimizer.voptlefty(iInitialFrames - 1)), 3, cv::Scalar(255, 255, 255));
                    circle(Tracker.debugImage, cv::Point( Optimizer.voptrightx(iInitialFrames - 1), Optimizer.voptrighty(iInitialFrames - 1)), 3, cv::Scalar(255, 255, 255));
                }
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
            if( (char)c == 'f' ) { imwrite("frame.png",frame); }
        }
    }

    Tracker.releaseCornerKernels();

    return 0;
}
