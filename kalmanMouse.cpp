#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <iomanip>
 
#include <eigen3/Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include "rapidcsv.h"
 
namespace Eigen{
  namespace internal{
    template<typename Scalar>
    struct scalar_normal_dist_op{
      static boost::mt19937 rng;    // The uniform pseudo-random algorithm
      mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

      EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

      template<typename Index>
      inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
    };

    template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng;
    template<typename Scalar>
    struct functor_traits<scalar_normal_dist_op<Scalar> >{ 
      enum{ Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false };
    };
  }
}
 
template<typename Clock, typename Duration>
std::ostream &operator<<(std::ostream &stream,  const std::chrono::time_point<Clock, Duration> &time_point){
  const time_t time = Clock::to_time_t(time_point);
  struct tm tm;
  localtime_r(&time, &tm);
  return stream << std::put_time(&tm, "%c"); // Print standard date&time
}
 
auto start = std::chrono::high_resolution_clock::now();
int k=3;
 
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

  std::vector<cv::Point> groundTruth, kalmanv, measurmens;
  groundTruth.reserve(50);
  kalmanv.reserve(50);
  measurmens.reserve(50);

  cv::Mat img(500, 500, CV_8UC3);
  cv::KalmanFilter KF(4, 2, 0);
  cv::Mat_<float> state(4, 1); // (x, y, Vx, Vy) 
  cv::Mat processNoise(4, 1, CV_32F);
  cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));
  char code = (char)-1;
  int skip_m = 6;
  int i = skip_m;

  cv::namedWindow("Mouse Tracker with Kalman Filter");
  cv::setMouseCallback("Mouse Tracker with Kalman Filter", on_mouse, nullptr);
  double delta_t=1/20.0;

  for(;;){
    if (mouse_info.x < 0 || mouse_info.y < 0) {
      imshow("Mouse Tracker with Kalman Filter", img);
      cv::waitKey(30);
      continue;
    }

    cv::Mat transitionMatrix=(cv::Mat_<float>(4, 4) << 1,0,delta_t,0,   0,1,0,delta_t,  0,0,1,0,  0,0,0,1);
    KF.transitionMatrix = transitionMatrix;

    setIdentity(KF.measurementMatrix);
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-0));
    cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(k*1e-0));
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.2));
    cv::setIdentity(KF.errorCovPre,cv::Scalar::all(.1));

    // measurmens.clear();
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

      // perform measurement every x loops
      if(i%skip_m == 0){
        // take new measurements+noise
        measurement(0) = mouse_info.x;
        measurement(1) = mouse_info.y;
        cv::Point newGroundTruth(mouse_info.x,mouse_info.y);
        groundTruth.push_back(newGroundTruth);
        measPt = cv::Point(measurement(0),measurement(1));
        measurmens.push_back(measPt);

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
      for (std::size_t i = 0; i < measurmens.size()-1; i++){
          line(img, measurmens[i], measurmens[i+1], cv::Scalar(0,255,255), 1);
      }

      if(groundTruth.size() > 300){
        measurmens.clear();
        kalmanv.clear();
        groundTruth.clear();
        i = skip_m;
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