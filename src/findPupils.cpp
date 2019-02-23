#include "findPupils.h"


PupilsTracker::PupilsTracker(){}

void PupilsTracker::Initializer()
{
    //-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
    face_cascade_name = "../../res/haarcascade_frontalface_alt.xml";

    // Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        exit(-1);
    }


    main_window_name = "Capture - Face detection";
    face_window_name = "Capture - Face";

    cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
    cv::moveWindow(main_window_name, 400, 100);

    // commented by zhantao deng@epfl, 23/02/2019
//    cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
//    cv::moveWindow(face_window_name, 10, 100);
}

/**
 * @function detector for pupils
 */
void PupilsTracker::DetectPupils(cv::Mat frame)
{
    std::vector<cv::Rect> faces;

    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];

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
    rectangle(debugImage, faces[i], cv::Scalar(255, 255, 255));
    }

    //-- Show what you got
    // if detected a face
    if (faces.size() > 0) {

        FindEyes(frame_gray, faces[0]);

//        circle(debugImage, left_eyepos, 3, cv::Scalar(255, 255, 255));
//        circle(debugImage, right_eyepos, 3, cv::Scalar(255, 255, 255));
    }
    else {
        printf("Face tracking lost\n");
    }
}



void PupilsTracker::FindEyes(cv::Mat frame_gray, cv::Rect face) {
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
  cv::Point leftPupil  = FindEyeCenter(faceROI,leftEyeRegion);
  cv::Point rightPupil = FindEyeCenter(faceROI,rightEyeRegion);
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

//  commented by zhantao deng@epfl, 23/02/2019
//  rectangle(debugFace,leftRightCornerRegion,200);
//  rectangle(debugFace,leftLeftCornerRegion,200);
//  rectangle(debugFace,rightLeftCornerRegion,200);
//  rectangle(debugFace,rightRightCornerRegion,200);

  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;

  // commented by zhantao deng@epfl, 23/02/2019
  // draw eye centers
//  circle(debugFace, rightPupil, 3, 1234);
//  circle(debugFace, leftPupil, 3, 1234);

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

// commented by zhantao deng@epfl, 23/02/2019, now we do not want to show the face
//  imshow(face_window_name, faceROI);

  left_eyepos  = cv::Point( leftPupil.x + face.x,  leftPupil.y + face.y );
  right_eyepos = cv::Point(rightPupil.x + face.x, rightPupil.y + face.y );
}


cv::Point PupilsTracker::FindEyeCenter(cv::Mat face, cv::Rect eye)
{
    cv::Mat eyeROIUnscaled = face(eye);
    cv::Mat eyeROI;
    scaleToFastSize(eyeROIUnscaled, eyeROI);

    // commented by zhantao deng@epfl, 23/02/2019
    // draw eye region
//    rectangle(face,eye,1234);

    //-- Find the gradient
    cv::Mat gradientX = computeMatXGradient(eyeROI);
    cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();
    //-- Normalize and threshold the gradient
    // compute all the magnitudes
    cv::Mat mags = matrixMagnitude(gradientX, gradientY);
    //compute the threshold
    double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);

    //normalize
    for (int y = 0; y < eyeROI.rows; ++y) {
      double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
      const double *Mr = mags.ptr<double>(y);
      for (int x = 0; x < eyeROI.cols; ++x) {
        double gX = Xr[x], gY = Yr[x];
        double magnitude = Mr[x];
        if (magnitude > gradientThresh) {
          Xr[x] = gX/magnitude;
          Yr[x] = gY/magnitude;
        } else {
          Xr[x] = 0.0;
          Yr[x] = 0.0;
        }
      }
    }

    //-- Create a blurred and inverted image for weighting
    cv::Mat weight;
    GaussianBlur( eyeROI, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
    for (int y = 0; y < weight.rows; ++y) {
      unsigned char *row = weight.ptr<unsigned char>(y);
      for (int x = 0; x < weight.cols; ++x) {
        row[x] = (255 - row[x]);
      }
    }

    //-- Run the algorithm!
    cv::Mat outSum = cv::Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
    // for each possible gradient location
    // Note: these loops are reversed from the way the paper does them
    // it evaluates every possible center for each gradient location instead of
    // every possible gradient location for every center.

    for (int y = 0; y < weight.rows; ++y) {
      const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
      for (int x = 0; x < weight.cols; ++x) {
        double gX = Xr[x], gY = Yr[x];
        if (gX == 0.0 && gY == 0.0) {
          continue;
        }
        testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
      }
    }
    // scale all the values down, basically averaging them
    double numGradients = (weight.rows*weight.cols);
    cv::Mat out;
    outSum.convertTo(out, CV_32F,1.0/numGradients);

    //-- Find the maximum point
    cv::Point maxP;
    double maxVal;
    cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
    //-- Flood fill the edges
    if(kEnablePostProcess) {
      cv::Mat floodClone;
      //double floodThresh = computeDynamicThreshold(out, 1.5);
      double floodThresh = maxVal * kPostProcessThreshold;
      cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
      if(kPlotVectorField) {
        //plotVecField(gradientX, gradientY, floodClone);
        cv::imwrite("eyeFrame.png",eyeROIUnscaled);
      }
      cv::Mat mask = floodKillEdges(floodClone);

      // redo max
      cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
    }
    return unscalePoint(maxP,eye);
}

bool inMat(cv::Point p,int rows,int cols) {
  return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

cv::Mat PupilsTracker::floodKillEdges(cv::Mat &mat)
{
  rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);

  cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
  std::queue<cv::Point> toDo;
  toDo.push(cv::Point(0,0));
  while (!toDo.empty()) {
    cv::Point p = toDo.front();
    toDo.pop();
    if (mat.at<float>(p) == 0.0f) {
      continue;
    }
    // add in every direction
    cv::Point np(p.x + 1, p.y); // right
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}

bool PupilsTracker::floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
  return inMat(np, mat.rows, mat.cols);
}


void PupilsTracker::testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
  // for all possible centers
  for (int cy = 0; cy < out.rows; ++cy) {
    double *Or = out.ptr<double>(cy);
    const unsigned char *Wr = weight.ptr<unsigned char>(cy);
    for (int cx = 0; cx < out.cols; ++cx) {
      if (x == cx && y == cy) {
        continue;
      }
      // create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double dotProduct = dx*gx + dy*gy;
      dotProduct = std::max(0.0,dotProduct);
      // square and multiply by the weight
      if (kEnableWeight) {
        Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
      } else {
        Or[cx] += dotProduct * dotProduct;
      }
    }
  }
}

cv::Point PupilsTracker::unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

void PupilsTracker::scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}


double PupilsTracker::computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor) {
  cv::Scalar stdMagnGrad, meanMagnGrad;
  cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
  double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
  return stdDevFactor * stdDev + meanMagnGrad[0];
}

cv::Mat PupilsTracker::matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY) {
  cv::Mat mags(matX.rows,matX.cols,CV_64F);
  for (int y = 0; y < matX.rows; ++y) {
    const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
    double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < matX.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = sqrt((gX * gX) + (gY * gY));
      Mr[x] = magnitude;
    }
  }
  return mags;
}

cv::Mat PupilsTracker::computeMatXGradient(const cv::Mat &mat) {
  cv::Mat out(mat.rows,mat.cols,CV_64F);

  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);

    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }

  return out;
}



/** for eye corner tracking **/

void PupilsTracker::createCornerKernels() {

    float kEyeCornerKernel[4][6] = {
      {-1,-1,-1, 1, 1, 1},
      {-1,-1,-1,-1, 1, 1},
      {-1,-1,-1,-1, 0, 3},
      { 1, 1, 1, 1, 1, 1},
    };

  rightCornerKernel = new cv::Mat(4,6,CV_32F, kEyeCornerKernel);
  leftCornerKernel = new cv::Mat(4,6,CV_32F);
  // flip horizontally and save to leftConerKernel
  cv::flip(*rightCornerKernel, *leftCornerKernel, 1);
}

cv::Point2f PupilsTracker::findEyeCorner(cv::Mat region, bool left, bool left2) {
  cv::Mat cornerMap = eyeCornerMap(region, left, left2);

  cv::Point maxP;
  cv::minMaxLoc(cornerMap,NULL,NULL,NULL,&maxP);

  cv::Point2f maxP2;
  maxP2 = findSubpixelEyeCorner(cornerMap, maxP);

  return maxP2;
}

cv::Point2f PupilsTracker::findSubpixelEyeCorner(cv::Mat region, cv::Point maxP) {

  cv::Size sizeRegion = region.size();

  cv::Mat cornerMap(sizeRegion.height * 10, sizeRegion.width * 10, CV_32F);

  cv::resize(region, cornerMap, cornerMap.size(), 0, 0, cv::INTER_CUBIC);

  cv::Point maxP2;
  cv::minMaxLoc(cornerMap, NULL,NULL,NULL,&maxP2);

  return cv::Point2f(sizeRegion.width / 2 + maxP2.x / 10,
                     sizeRegion.height / 2 + maxP2.y / 10);
}

cv::Mat PupilsTracker::eyeCornerMap(const cv::Mat &region, bool left, bool left2) {
  cv::Mat cornerMap;

  cv::Size sizeRegion = region.size();
  cv::Range colRange(sizeRegion.width / 4, sizeRegion.width * 3 / 4);
  cv::Range rowRange(sizeRegion.height / 4, sizeRegion.height * 3 / 4);

  cv::Mat miRegion(region, rowRange, colRange);

  cv::filter2D(miRegion, cornerMap, CV_32F,
               (left && !left2) || (!left && !left2) ? *leftCornerKernel : *rightCornerKernel);

  return cornerMap;
}

void  PupilsTracker::releaseCornerKernels() {
  delete leftCornerKernel;
  delete rightCornerKernel;
}
