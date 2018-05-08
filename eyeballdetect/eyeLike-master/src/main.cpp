#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <fstream>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"


/** Constants **/


/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "C://opencv//data//haarcascades//haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);
std::ofstream outputfileleft, outputfileright, Output;

/*
int left_detectEye(cv::Mat& org, cv::Mat& im, cv::Mat& tpl, cv::Rect& rect, cv::Rect& faces, cv::Mat& face, int k)
{
	std::vector<cv::Rect> eyes;
	cv::Rect leftside(0, 0, face.rows/2, face.cols);

	for ( int i = 0; i < k; i++)
	{
		left_eye_cascade.detectMultiScale(face(leftside), eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20,20));
		if (eyes.size())
		{
			
			rect = eyes[0] + cv::Point(faces.x, faces.y);
			tpl  = im(rect);
			leftcroppedEyeImage = org (rect);
			
			//tpl = face(eyes[i]);
			
		}
	}

	return eyes.size();
}

int right_detectEye(cv::Mat& org, cv::Mat& im, cv::Mat& tpl, cv::Rect& rect, cv::Rect& faces, cv::Mat& face, int k)
{
	std::vector<cv::Rect>  eyes;
	cv::Rect rightside(face.rows/2, 0, face.rows/2, face.cols);
	
	for (int i = 0; i < k; i++)
	{
		right_eye_cascade.detectMultiScale(face(rightside), eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20,20));
		if (eyes.size())
		{
			
			rect = eyes[0]+ cv::Point(face.rows/2, 0) + cv::Point(faces.x, faces.y);
			tpl  = im(rect);
			rightcroppedEyeImage = org (rect);
			
			//tpl = face(eyes[i]);
			
		}
	}

	return eyes.size();
}
*/



/**
 * @function main
 */
int main( int argc, const char** argv ) {
  CvCapture* capture;
  cv::Mat frame;
  outputfileleft.open("dataleft.txt");
  outputfileright.open("dataright.txt");
  Output.open("Output Data.txt");

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  Output<<"Xml File Loaded"<<std::endl;
  Output<<"Video Starting"<<std::endl;
  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 400, 100);
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(face_window_name, 10, 100);
  
  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

   // Read the video stream
  capture = cvCaptureFromCAM( -1 );
  if( capture ) {
    while( true ) {
		//cvWaitKey(2000);
      frame = cvQueryFrame( capture );
	  Output<<frame.cols<<"\t"<<frame.rows<<std::endl;
	  
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
	  imwrite("face.jpg",debugImage);

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
  
  

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  
  Output<<"Eye Region Croped"<<std::endl;
  Output<<leftPupil<<"\t"<<rightPupil<<std::endl;
  
  cv::imwrite("leftEye.jpg", faceROI(leftEyeRegion));
  cv::imwrite("rightEye.jpg", faceROI(rightEyeRegion));

  int rightmovex= rightPupil.x*100/rightEyeRegion.width;
  int rightmovey= rightPupil.y*100/rightEyeRegion.height;
  int leftmovex= leftPupil.x*100/leftEyeRegion.width;
  int leftmovey= leftPupil.y*100/leftEyeRegion.height;

  //store eye position in data file
  outputfileright << rightmovex <<"\t"<< rightmovey << std::endl;
  outputfileleft << leftmovex<<"\t"<<leftmovey << std::endl;

  
  //declaring the derection to look.
  if(rightmovex < 52)
  	  Output<< "Right  left"<<std::endl;
  else if(rightmovex > 54)
	  Output<<"Right   Right"<<std::endl;
  else
	  Output<<"Right  Center"<<std::endl;


  if(leftmovex < 47)
	  Output<< "Left  left"<<std::endl;
  else if(leftmovex > 51 )
	  Output<<"Left    Right"<<std::endl;
  else
	  Output<< "Left   Center"<<std::endl;
	  


  /*
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
  */


  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  
  
  // draw eye centers
  circle(debugFace, rightPupil, 3, 1234);
  circle(debugFace, leftPupil, 3, 1234);

  //if (rightPupil.x - 
    

  /*
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
  */

  imshow(face_window_name, faceROI);
  imwrite("FaceImage.jpg", faceROI);
  //cvCreateVideoWriter("video.avi", CV_FOURCC_DEFAULT, 15.0, cv::Size(faceROI.rows, faceROI.cols), 0); 
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
  cv::Mat frame_gray;

  //std::vector<cv::Mat> rgbChannels(3);
  //cv::split(frame, rgbChannels);
  //cv::Mat frame_gray = rgbChannels[2];

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );

  for( int i = 0; i < faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], 1234);
  }
  //-- Show what you got
  if (faces.size() > 0) {
    findEyes(frame_gray, faces[0]);
  }
}
