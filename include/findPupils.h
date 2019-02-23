#ifndef FINDPUPILS_H
#define FINDPUPILS_H

#include <queue>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>

#include "constants.h"

class PupilsTracker
{
public:
    PupilsTracker();

    void Initializer();

    void createCornerKernels();

    void DetectPupils( cv::Mat frame );

    void FindEyes(cv::Mat frame_gray, cv::Rect face);

    cv::Point FindEyeCenter(cv::Mat face, cv::Rect eye);

    cv::Point2f findEyeCorner(cv::Mat region, bool left, bool left2);

    void releaseCornerKernels();

protected:
    cv::Mat floodKillEdges(cv::Mat &mat);
    bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat);

    cv::Point unscalePoint(cv::Point p, cv::Rect origSize);
    void scaleToFastSize(const cv::Mat &src,cv::Mat &dst);
    double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);
    cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
    cv::Mat computeMatXGradient(const cv::Mat &mat);

    void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out);

    cv::Mat eyeCornerMap(const cv::Mat &region, bool left, bool left2);
    cv::Point2f findSubpixelEyeCorner(cv::Mat region, cv::Point maxP);

    cv::CascadeClassifier face_cascade;

    cv::Mat *leftCornerKernel;
    cv::Mat *rightCornerKernel;


public:

    cv::Mat debugImage;

    cv::String face_cascade_name;
    std::string main_window_name;
    std::string face_window_name;

    cv::Point left_eyepos, right_eyepos;
};

#endif
