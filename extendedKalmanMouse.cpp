#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>

#include <eigen3/Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
 
auto start = std::chrono::high_resolution_clock::now();
int k=1;

struct mouse_info_struct { int x,y; };
struct mouse_info_struct mouse_info = {-1,-1}, last_mouse;
 
void on_mouse(int event, int x, int y, int flags, void* param) {
  last_mouse = mouse_info;
  mouse_info.x = x;
  mouse_info.y = y;
}
 
// plot points
#define drawCross( center, color, d )                                 \
cv::line( img, cv::Point( center.x - d, center.y - d ),                \
cv::Point( center.x + d, center.y + d ), color, 2, cv::LINE_AA, 0); \
cv::line( img, cv::Point( center.x + d, center.y - d ),                \
cv::Point( center.x - d, center.y + d ), color, 2, cv::LINE_AA, 0 )
 
 
int main (int argc, char * const argv[]) {

  std::vector<cv::Point> groundTruth, kalmanv, measurements;
  groundTruth.reserve(50);
  kalmanv.reserve(50);
  measurements.reserve(50);

  cv::Mat img(500, 500, CV_8UC3);
  cv::KalmanFilter KF(4, 2, 0); // (x, y, Vx, Vy) 
  cv::Mat_<float> state(4, 1); 
  cv::Mat processNoise(4, 1, CV_32F);
  cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));
  char code = (char)-1;
  int i = 9;

  cv::namedWindow("Mouse Tracker with Kalman Filter");
  cv::setMouseCallback("Mouse Tracker with Kalman Filter", on_mouse, nullptr);
  double delta_t=1/20.0;

  for(;;){
    if (mouse_info.x < 0 || mouse_info.y < 0) {
      imshow("Mouse Tracker with Kalman Filter", img);
      cv::waitKey(30);
      continue;
    }

    cv::Mat transitionMatrix
      =(cv::Mat_<float>(4, 4) << 1,0,delta_t,0, 
                                 0,1,0,delta_t,
                                 0,0,1,0,  
                                 0,0,0,1);
    KF.transitionMatrix = transitionMatrix;

    setIdentity(KF.measurementMatrix);
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(5e-1));     // Q, higher = less filtered, faster response
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-2)); // R, higher = more filtered, slower response
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.2));
    cv::setIdentity(KF.errorCovPre,cv::Scalar::all(.1));

    // measurements.clear();
    // groundTruth.clear();
    // kalmanv.clear();
    std::cout<< "measurementMatrix"<<std::endl;
    std::cout<<KF.measurementMatrix<<std::endl;

    cv::Point statePt = cv::Point(0, 0);
    cv::Point measPt = cv::Point(0, 0);
    cv::Mat estimated;

    for(;;){

      cv::Mat prediction = KF.predict();
      cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

      // perform measurement every three loops
      if(i%9 == 0){
        // take new measurements
        measurement(0) = mouse_info.x;
        measurement(1) = mouse_info.y;
        cv::Point newGroundTruth(mouse_info.x,mouse_info.y);
        groundTruth.push_back(newGroundTruth);
        measPt = cv::Point(measurement(0),measurement(1));
        measurements.push_back(measPt);

        estimated = KF.correct(measurement);
        statePt = cv::Point(estimated.at<float>(0), estimated.at<float>(1));
        kalmanv.push_back(statePt);
      }else{
        // take measurement for display purposes only, do not give to kf
        cv::Point newGroundTruth(mouse_info.x,mouse_info.y);
        groundTruth.push_back(newGroundTruth);

        //http://opencv-users.1802565.n2.nabble.com/Kalman-filters-and-missing-measurements-td2886593.html
        KF.errorCovPre.copyTo(KF.errorCovPost);
        KF.statePre.copyTo(KF.statePost);
      }

      i++;

      img = cv::Scalar::all(0);
      drawCross( statePt, cv::Scalar(255,255,255), 5 );
      drawCross( measPt, cv::Scalar(0,0,255), 5 );
      drawCross( predictPt, cv::Scalar(255,0,255), 5 );

      for (std::size_t i = 0; i < groundTruth.size()-1; i++){
          line(img, groundTruth[i], groundTruth[i+1], cv::Scalar(0,255,0), 1);
      }
      for (std::size_t i = 0; i < kalmanv.size()-1; i++){
          line(img, kalmanv[i], kalmanv[i+1], cv::Scalar(255,0,0), 1);
      }
      for (std::size_t i = 0; i < measurements.size()-1; i++){
          line(img, measurements[i], measurements[i+1], cv::Scalar(0,255,255), 1);
      }

      if(groundTruth.size() > 300){
        measurements.clear();
        kalmanv.clear();
        groundTruth.clear();
        i = 9;
      }

      int line_text = 10;
      cv::putText(img,"- Measurements + noise", cv::Point(10,line_text), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,255));
      cv::putText(img,"- Ground truth", cv::Point(10,line_text += 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
      cv::putText(img,"- Kalman state", cv::Point(10,line_text += 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,0));
      cv::putText(img,"X = State", cv::Point(10,line_text += 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255));
      cv::putText(img,"X = Measured", cv::Point(10,line_text += 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255));
      cv::putText(img,"X = Predicted", cv::Point(10,line_text += 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,0,255));
      imshow( "Mouse Tracker with Kalman Filter", img );
      code = (char)cv::waitKey(1000.0*delta_t);

      if( code == 27 || code == 'q' || code == 'Q' || code == 33 ){
        break;
      }

    }

    if( code == 27 || code == 'q' || code == 'Q' || code == 33 ){
      break;
    }
    
  }
  return 0;
}