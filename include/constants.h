#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ********************* Paras for pupils tracking ************************ */
// Debugging
const bool kPlotVectorField = false;

// Size constants
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;

// Preprocessing
const bool kSmoothFaceImage = false;
const float kSmoothFaceFactor = 0.005;

// Algorithm Parameters
const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const bool kEnableWeight = true;
const float kWeightDivisor = 1.0;
const double kGradientThreshold = 50.0;

// Postprocessing
const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.97;

// Eye Corner
const bool kEnableEyeCorner = false;


/* ******************** Paras for eyepose optimizing ********************** */
// Average radius of eyeballs
const float fEyeRadius = 0.012; // in meters

// Optimum eye rotation
/* "Optimum" means:
 * "placement of objects in one's visual field for best recognition should take into account that optimum eye rotation"
 * ref: [1] Human Dimension & Interior Space, 1979.
 *      [2] http://www.hazardcontrol.com/factsheets/humanfactors/visual-acuity-and-line-of-sight
 * Here we assume that people prefre to place screen in their optimal field
 */
const bool bUseOptField = true;
const int iOptUpward    = 15;  // in degree
const int iOptDownward  = 15;
const int iOptLeft      = 15;
const int iOptRight     = 15;
const int iMaxUpward    = 25;
const int iMaxDownward  = 30;
const int iMaxLeft      = 35;
const int iMaxRight     = 35;

// Interpupillary distance
// ref: Interpupillary Distance Measurements among Students in the Kumasi Metropolis
const float fInterPupilDist = 0.063; // general in [ 59 - 67 mm]

// Binking time
// ref: https://core.ac.uk/download/pdf/37456352.pdf
const float fBlinkFreq = 20.0;    // times per minutes, during conversation: 3-5 seconds; reading/watching: 5-10 seconds;
const float fBlinkTime = 0.3;     // time for blinking, in second
const float fVanishTime   = fBlinkTime/2;   // loss only in a part of time

// Initiatial distance between camera and face will be updated
const float fInitCamFaceDist = 0.65;

// Parameters that still need supports from researches
const float fMinBackforth = 0.2;   // seconds for returning move, my eyes, if continuously move the max freq should be lower I guess
const float fAvgEyeSpeed  = 0.5;   // my eyes, when I focus on something.

// Initial frames for streaming
const int iInitialFrames = 15;

// Max iterations
const int iMaxiters = 100;

// Regularization parameter
const float fGama = 5.0;  // regularization parameter
const float fGamaStr = 1000.0;

// Learning rate for eye pose and screen-face distance
// -- for GD, all data optimized together
const float fLearnRatePose = 0.01;     // in pixel
const float fLearnRateDist = 0.00001; // in meter
// -- for SGD and streaming optimization
const double fLearnRatePoseStr = 0.0002;    // in pixel
const double fLearnRateDistStr = 0.0000002; // in meter

#endif
