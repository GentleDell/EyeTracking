#ifndef OPTEYEPOSE_H
#define OPTEYEPOSE_H

#include <stdio.h>
#include <vector>
#include <iostream>

#include "eigen3/Eigen/Core"
#include "opencv2/imgproc/imgproc.hpp"
#include "constants.h"

using namespace std;

struct Gradient
{
    Eigen::VectorXd Grad_lx, Grad_ly, Grad_rx, Grad_ry;

    Eigen::VectorXd Grad_dist;
};


class SGDOptimizer
{
public:
    SGDOptimizer(double f);

    void receiver(cv::Point pLeftEyePosition, cv::Point pRightEyePosition);

    // initialize the class with given point vectors
    void Initialize( vector<cv::Point> vLeftEyePosition, vector<cv::Point> vRightEyePosition , double f);

    // filter outliers in the streamed data
    //     since there are different thresholds, the bool flag is needed
    void MedianFilter( Eigen::VectorXd &vEyePosition, bool horizontal );

    void JointOptimize();

    void ComputeLoss();

    void ComputeGrad();

    void optimize();

    // append new measurements to the eyepose vector
    void update( cv::Point pLeftEyePosition, cv::Point pRightEyePosition );



public:
    bool initiated;

    double focal;

    double Loss;

    Eigen::VectorXd vErrors;

    std::vector<double> vLoss;

    Gradient svGradient;

    float ThresDiffUpDown, ThresDiffLefRig;

    Eigen::MatrixXd updateMat, diffMat;  // matrix to update and differentiate pose vector

    Eigen::VectorXd vScFaceDist;
    Eigen::VectorXd vleftx, vlefty, vrightx, vrighty;
    Eigen::VectorXd voptleftx, voptlefty, voptrightx, voptrighty;

    std::vector<cv::Point> vInit_LeftEyePosition, vInit_RightEyePosition;

};


#endif
