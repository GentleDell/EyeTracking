#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"


/** Constants **/


/** Function Headers */
void detectAndDisplay( cv::Mat frame );
void read_param(cv::Mat & camera_Mat, cv::Mat & distCoeffs, cv::Size imageSize, const char * IntrinsicsPath);

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
static cv::String face_cascade_name = "../../res/haarcascade_frontalface_alt.xml";
static cv::CascadeClassifier face_cascade;
static std::string main_window_name = "Capture - Face detection";
static std::string face_window_name = "Capture - Face";
static cv::RNG rng(12345);
static cv::Mat debugImage;
static cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

/**
 * @function main
 */
int main( int argc, const char** argv ) {
  cv::Mat frame, undist_frame;
  int64 t0 = cv::getTickCount();
  size_t nFrames = 1;

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };

  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 400, 100);
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(face_window_name, 10, 100);
//  commented by zhantao deng @epfl 03/02/2019
//  cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
//  cv::moveWindow("Right Eye", 10, 600);
//  cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
//  cv::moveWindow("Left Eye", 10, 800);

  // create eye-coner kernel
  createCornerKernels();
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
      frame.copyTo(debugImage);

      // Apply the classifier to the frame
      if( !frame.empty() ) {
        detectAndDisplay( frame );
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      imshow(main_window_name,debugImage);

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

  releaseCornerKernels();

  return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
  cv::Mat faceROI = frame_gray(face);
  cv::Mat debugFace = faceROI;

  // Smoothen teh face image using Gaussian filter, default NOT
  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);     // eye width
  int eye_region_height = face.width * (kEyePercentHeight/100.0);   // eye height
  int eye_region_top = face.height * (kEyePercentTop/100.0);        // top left conner y
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  cv::Point leftPupil  = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  // get corner regions
  cv::Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  cv::Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  cv::Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  cv::Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);

  //-- Find Eye Corners
  if (kEnableEyeCorner) {
    cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
    cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
    cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
    cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
    circle(faceROI, leftRightCorner, 3, 200);
    circle(faceROI, leftLeftCorner, 3, 200);
    circle(faceROI, rightLeftCorner, 3, 200);
    circle(faceROI, rightRightCorner, 3, 200);
  }

  imshow(face_window_name, faceROI);

  // save pupil pose for filtering, by zhantao deng @ EPFL, 30/01/2019
  int64 time = cv::getTickCount();
  std::ofstream ofs;
  ofs.open ("pupil_pose.txt", std::ofstream::out | std::ofstream::app);
  ofs << (double)time * 1000.0f/cv::getTickFrequency() << ',' << leftPupil.x + face.x << ',' << leftPupil.y + face.y << ',' << rightPupil.x + face.x  << ',' << rightPupil.y + face.y << std::endl;
  ofs.close();

//  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
//  cv::Mat destinationROI = debugImage( roi );
//  faceROI.copyTo( destinationROI );
}


cv::Mat findSkin (cv::Mat &frame) {
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
//    uchar *Or = output.ptr<uchar>(y);
    cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      cv::Vec3b ycrcb = Mr[x];
//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = cv::Vec3b(0,0,0);
      }
    }
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) {
  std::vector<cv::Rect> faces;
  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];

  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);

  //-- Detect faces
  /*
    frame_gray:     Matrix of the type CV_8U containing an image where objects are detected.
    faces:          Vector of rectangles where each rectangle contains the detected faces, the rectangles may be partially outside the original image.
    scaleFactor:    Parameter specifying how much the image size is reduced at each image scale.
    minNeighbors:   specifying how many neighbors each candidate rectangle should have to retain it.
    flags:          with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
    minSize:        possible object size. Objects smaller than that are ignored.
    maxSize:        Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
   */
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );


  //-- findSkin(debugImage);
  for( int i = 0; i < faces.size(); i++ )
  {
    // draw a rectangle scaling by face[i], color 1234
    rectangle(debugImage, faces[i], 1234);
  }


  //-- Show what you got
  //
  if (faces.size() > 0) {
      findEyes(frame_gray, faces[0]);
  }
}

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
