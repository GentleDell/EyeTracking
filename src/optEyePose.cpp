#include "optEyePose.h"

#define sind(x) (sin(fmod((x),360) * M_PI / 180))

/*
 * Constructor of SGD optimizor
 */
SGDOptimizer::SGDOptimizer(double f){
    focal = f;
    initiated = false;
}

void SGDOptimizer::receiver(cv::Point pLeftEyePosition, cv::Point pRightEyePosition)
{
    if (vInit_LeftEyePosition.size() < iInitialFrames && !initiated){

        vInit_LeftEyePosition.push_back(pLeftEyePosition);
        vInit_RightEyePosition.push_back(pRightEyePosition);
    }
    else if (vInit_LeftEyePosition.size() == iInitialFrames && !initiated)
    {
        Initialize(vInit_LeftEyePosition, vInit_RightEyePosition, focal);
        vInit_LeftEyePosition.clear();
        vInit_RightEyePosition.clear();

        optimize();
    }
    else
    {
        update(pLeftEyePosition, pRightEyePosition);
        optimize();
    }
}

/*
 * Initialize eyepose data and thresholds for the median filter
 * Set the update Matrix
 */
void SGDOptimizer::Initialize(vector<cv::Point> vLeftEyePosition, vector<cv::Point> vRightEyePosition, double f )
{
    initiated = true;

    vleftx  = Eigen::VectorXd::Zero(iInitialFrames);
    vlefty  = Eigen::VectorXd::Zero(iInitialFrames);
    vrightx = Eigen::VectorXd::Zero(iInitialFrames);
    vrighty = Eigen::VectorXd::Zero(iInitialFrames);

    // initialize pose data
    for (int ct = 0; ct < iInitialFrames; ct++)
    {
     vleftx(ct) = vLeftEyePosition[ct].x;
     vlefty(ct) = vLeftEyePosition[ct].y;

     vrightx(ct) = vRightEyePosition[ct].x;
     vrighty(ct) = vRightEyePosition[ct].y;
    }

    // initialize thresholds for median filter
    if(bUseOptField)
    {
     // assume screen is inside the optimal vision field (explained in the load_parameters.m)
     ThresDiffUpDown = 2*fEyeRadius*sind(iOptUpward)*f/fInitCamFaceDist;
     ThresDiffLefRig = 2*fEyeRadius* sind(iOptLeft) *f/fInitCamFaceDist;
    }
    else
    {
     // assume screen is inside the maximal vision field (explained in the load_parameters.m)
     ThresDiffUpDown = fEyeRadius*( sind(iMaxUpward) + sind(iMaxDownward) )*f/fInitCamFaceDist;
     ThresDiffLefRig = 2*fEyeRadius* sind(iMaxLeft) *f/fInitCamFaceDist;
    }

    // focal length and screen-face distance
    focal = f;
    vScFaceDist = fInitCamFaceDist * Eigen::VectorXd::Ones(iInitialFrames);

    // Matrix to update pose data
    //     Superdiagnoal entries and the last entry are set to be 1
    updateMat =  Eigen::MatrixXd::Zero(iInitialFrames,iInitialFrames);
    updateMat.diagonal(1) = Eigen::VectorXd::Ones(iInitialFrames - 1);
    updateMat(iInitialFrames - 1, iInitialFrames - 1) = 1;

    // Matrix to compute the difference of pose vector
    diffMat = - Eigen::Matrix<double, iInitialFrames-1, iInitialFrames>::Identity();
    diffMat.diagonal(1) = Eigen::VectorXd::Ones(iInitialFrames - 1);
}

/*
 * pop the first entry then push the new data in
 */
void SGDOptimizer::update(cv::Point pLeftEyePosition, cv::Point pRightEyePosition)
{
    vleftx = updateMat*vleftx;
    vlefty = updateMat*vlefty;
    vleftx(iInitialFrames - 1) = pLeftEyePosition.x;
    vlefty(iInitialFrames - 1) = pLeftEyePosition.y;

    vrightx = updateMat*vrightx;
    vrighty = updateMat*vrighty;
    vrightx(iInitialFrames - 1) = pRightEyePosition.x;
    vrighty(iInitialFrames - 1) = pRightEyePosition.y;

    vScFaceDist = updateMat*vScFaceDist;
    vScFaceDist(iInitialFrames - 1) = fInitCamFaceDist;
}

/*
 * Median Filter:
 *      recover outliers using median value.
 * For data in the window, if the difference between a data and the median
 * value is larger than the given threshold, the data will be treated as an
 * outlier and be replaced with median value in the window.
 */
void SGDOptimizer::MedianFilter( Eigen::VectorXd &vEyePosition, bool horizontal )
{
    double max_data = 0, min_data = 0, means = 0;
    Eigen::VectorXd diff_eyepose;

    // checking horizontal data
    if (horizontal){

        max_data = vEyePosition.maxCoeff();
        min_data = vEyePosition.minCoeff();
        means = vEyePosition.mean();

        vEyePosition = ( (vEyePosition.array() - means) >= ThresDiffLefRig ).select(min_data, vEyePosition);
        vEyePosition = ( (vEyePosition.array() - means) <= -ThresDiffLefRig ).select(max_data, vEyePosition);
    }

    // checking vertical data
    else {
        max_data = vEyePosition.maxCoeff();
        min_data = vEyePosition.minCoeff();
        means = vEyePosition.mean();

        vEyePosition = ( (vEyePosition.array() - means) >= ThresDiffUpDown ).select(min_data, vEyePosition);
        vEyePosition = ( (vEyePosition.array() - means) <= -ThresDiffUpDown ).select(max_data, vEyePosition);
    }
}

/*
 * Compute Loss:
 *      compute the loss function and errors caused by perspective geometry.
 * The loss function involves three part: perspective geometry, noises and 1-norm
 * regularization. The 1 norm regularization term is implemented on the first order
 * derivatives of eye poses to denoise, which performs like a Total Variation term
 * for an image.
 */
void SGDOptimizer::ComputeLoss()
{
    double Loss_measure, Loss_reg;
    Eigen::VectorXd delta_x, delta_y;

    // Errors from perspective geometry
    // '*' here is element wise
    vErrors = ( (voptleftx - voptrightx).array().pow(2) + (voptlefty - voptrighty).array().pow(2) ) * vScFaceDist.array().pow(2) - std::pow(focal,2) * std::pow(fInterPupilDist, 2) ;

    // Loss from measurements noise
    Loss_measure = (voptleftx  -  vleftx).dot(voptleftx - vleftx) +
                   (voptlefty  -  vlefty).dot(voptlefty - vlefty) +
                   (voptrightx -  vrightx).dot(voptrightx - vrightx) +
                   (voptrighty -  vrighty).dot(voptrighty - vrighty) ;

    // Loss from regularization terms
    Loss_reg = fGamaStr * ( (diffMat * voptleftx).array().abs().sum() + (diffMat * voptrightx).array().abs().sum() +
                            (diffMat * voptlefty).array().abs().sum() + (diffMat * voptrighty).array().abs().sum());

    Loss = (vErrors.dot(vErrors) + Loss_measure) / iInitialFrames + Loss_reg;
    vErrors = vErrors.array()/iInitialFrames;
}

/* compute_gradient:
 *      compute gradients of eye poses and screen-face distance
 */
void SGDOptimizer::ComputeGrad()
{
    double eps = 1e-10;
    Eigen::VectorXd temp_head0 = Eigen::VectorXd::Zero(iInitialFrames),
                    temp_end0  = Eigen::VectorXd::Zero(iInitialFrames);

    Eigen::VectorXd squre_dist = vScFaceDist.array().pow(2);

    // transfer to array and conduct elementwise production  --  might have bugs !!!!!
    svGradient.Grad_dist = 2 * (vErrors.array() * ( 2*( (voptleftx - voptrightx).array().pow(2) + (voptlefty - voptrighty).array().pow(2) )) ) * vScFaceDist.array();

    temp_head0.tail(iInitialFrames - 1) = (diffMat * voptleftx).array() / ( (diffMat * voptleftx).array().abs() + eps);
    temp_end0.head(iInitialFrames - 1)  = (diffMat * voptleftx).array() / ( (diffMat * voptleftx).array().abs() + eps);
    svGradient.Grad_lx   = 2*vErrors.array() * ( 2*( voptleftx - voptrightx ).array() * squre_dist.array() ) + 2*( voptleftx - vleftx ).array()/iInitialFrames +
                           fGamaStr * temp_head0.array() - fGamaStr * temp_end0.array();

    temp_head0.tail(iInitialFrames - 1) = (diffMat * voptrightx).array() / ( (diffMat * voptrightx).array().abs() + eps);
    temp_end0.head(iInitialFrames - 1)  = (diffMat * voptrightx).array() / ( (diffMat * voptrightx).array().abs() + eps);
    svGradient.Grad_rx   = 2*vErrors.array() * (-2*( voptleftx - voptrightx ).array() * squre_dist.array() ) + 2*( voptrightx - vrightx ).array()/iInitialFrames +
                           fGamaStr * temp_head0.array() - fGamaStr * temp_end0.array();

    temp_head0.tail(iInitialFrames - 1) = (diffMat * voptlefty).array() / ( (diffMat * voptlefty).array().abs() + eps);
    temp_end0.head(iInitialFrames - 1)  = (diffMat * voptlefty).array() / ( (diffMat * voptlefty).array().abs() + eps);
    svGradient.Grad_ly   = 2*vErrors.array() * ( 2*( voptlefty - voptrighty ).array() * squre_dist.array() ) + 2*( voptlefty - vlefty ).array()/iInitialFrames +
                           fGamaStr * temp_head0.array() - fGamaStr * temp_end0.array();

    temp_head0.tail(iInitialFrames - 1) = (diffMat * voptrighty).array() / ( (diffMat * voptrighty).array().abs() + eps);
    temp_end0.head(iInitialFrames - 1)  = (diffMat * voptrighty).array() / ( (diffMat * voptrighty).array().abs() + eps);
    svGradient.Grad_ry   = 2*vErrors.array() * (-2*( voptlefty - voptrighty ).array() * squre_dist.array() ) + 2*( voptrighty - vrighty).array()/iInitialFrames +
                           fGamaStr * temp_head0.array() - fGamaStr * temp_end0.array();
}

/*
 * joint_filter:
 *      jointly optimize screen-face distances and eyeposes
 * using gradient descent method to iteratively optimize
 * eyepose and screen-face distance, with a joint cost funtion.
 */
void SGDOptimizer::JointOptimize()
{
    int   ct = 0;    // iteration counter
    double learn_rate_pose, learn_rate_dist;

    // initialize optimzied pose data
    voptleftx  = vleftx;
    voptlefty  = vlefty;
    voptrightx = vrightx;
    voptrighty = vrighty;

    for ( ct = 0; ct <= iMaxiters; ct ++)
    {
        // implement adaptive learning rate to achieve better solution. Using
        // log function as the decaying factor.
        learn_rate_pose = fLearnRatePoseStr/(log(ct+exp(1)));
        learn_rate_dist = fLearnRateDistStr/(log(ct+exp(1)));

        // compute loss
        ComputeLoss();

        // compute gradients of eye pose and screen-face distance
        ComputeGrad();

        // update eye pose and screen-face distance
        voptleftx  = voptleftx - learn_rate_pose * svGradient.Grad_lx;
        voptlefty  = voptlefty - learn_rate_pose * svGradient.Grad_ly;
        voptrightx = voptrightx - learn_rate_pose * svGradient.Grad_rx;
        voptrighty = voptrighty - learn_rate_pose * svGradient.Grad_ry;
        vScFaceDist = vScFaceDist - learn_rate_dist * svGradient.Grad_dist;

        // record total loss for visualization
        vLoss.push_back(Loss);

        // if the differences between loss_t and loss t-1 are less than 0.005,
        // the loss will be treated as converged.
        if (ct >= 100 && abs(vLoss.end()[-2] - vLoss.back()) <= 0.005)
        {
            std::cout << "----> The loss has converged" << std::endl;
            break;
        }
    }
}


void SGDOptimizer::optimize()
{
    bool horizontal = true;

    // clear all data in the vector -- prepare for the coming optimization loop
    vLoss.clear();

    // median filter to recover outliers
    MedianFilter( vleftx,  horizontal);
    MedianFilter( vrightx, horizontal);
    MedianFilter( vlefty,  !horizontal);
    MedianFilter( vrighty, !horizontal);

    // joint optimization
    JointOptimize();
}
