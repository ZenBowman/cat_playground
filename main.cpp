#include <SDL.h>
#include <algorithm>
#include <iostream>
#if defined(WINAPI_FAMILY) && (WINAPI_FAMILY == WINAPI_FAMILY_APP)
// On UWP, we need to not have SDL_main otherwise we'll get a linker error
#define SDL_MAIN_HANDLED
#endif
#include <SDL3_ttf/SDL_ttf.h>
#include <SDL_main.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

static bool app_quit = false;
const char *image_path = "gingertail_runwalk.bmp";

using ::cv::CHAIN_APPROX_SIMPLE;
using ::cv::Mat;
using ::cv::Point;
using ::cv::RETR_CCOMP;
using ::cv::Scalar;
using ::cv::Vec4i;
using ::cv::VideoCapture;

struct Pixel {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
  unsigned char hue;
  unsigned char saturation;
  unsigned char value;
};

const int winWidth = 1200;
const int winHeight = 1200;

enum Action { NONE, MOVE_LEFT, MOVE_RIGHT };

Mat draw_contours(Mat &src, int &maxAreaOut, Point &maxCenterOfMassOut,
                  int minArea) {
  Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
  std::vector<std::vector<Point>> contours;
  std::vector<Vec4i> hierarchy;

  findContours(src, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

  maxAreaOut = 0;
  int maxAreaIndex = 0;
  int currentArea;
  std::vector<int> indicesToDraw;
  std::vector<cv::Point> centerOfMasses;

  for (int i = 0; i < contours.size(); i++) {
    auto contour = contours[i];
    currentArea = contourArea(contour);

    if (currentArea > minArea) {
      indicesToDraw.push_back(i);
      cv::Point point;
      auto m = moments(contour);
      point.x = m.m10 / m.m00;
      point.y = m.m01 / m.m00;
      centerOfMasses.push_back(point);
    }

    if (currentArea > maxAreaOut) {
      // SDL_Log("Area found: %i", currentArea);
      maxAreaOut = currentArea;
      maxAreaIndex = i;
    }
  }

  if (centerOfMasses.size() == 0) {
    maxCenterOfMassOut.x = 0;
    maxCenterOfMassOut.y = 0;
  } else {
    maxCenterOfMassOut.x = 0;
    maxCenterOfMassOut.y = 0;
  }
  for (auto point : centerOfMasses) {
    maxCenterOfMassOut.x += point.x / centerOfMasses.size();
    maxCenterOfMassOut.y += point.y / centerOfMasses.size();
  }

  Scalar color = Scalar(255, 0, 0);
  for (auto i : indicesToDraw) {
    drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
  }

  return dst;
}

// Select green pixels from the camera frame.
Mat apply_green_filter(Mat &source, int min_green) {
  Mat hsvSource;
  cv::cvtColor(source, hsvSource, cv::COLOR_BGR2HSV);
  Mat newMatrix = Mat::zeros(source.rows, source.cols, CV_8UC1);
  const int rows = source.rows;
  const int cols = source.cols;

  const unsigned char *srcRowPointer;
  const unsigned char *hsvSrcRowPointer;
  unsigned char *destRowPointer;

  // Loop variables
  Pixel pixel;
  int srcColIndex;

  for (int i = 0; i < rows; i++) {
    srcRowPointer = source.ptr<uchar>(i);
    hsvSrcRowPointer = hsvSource.ptr<uchar>(i);
    destRowPointer = newMatrix.ptr<uchar>(i);
    for (int j = 0; j < cols; j++) {
      srcColIndex = j * 3;
      pixel.blue = srcRowPointer[srcColIndex];
      pixel.green = srcRowPointer[srcColIndex + 1];
      pixel.red = srcRowPointer[srcColIndex + 2];
      pixel.hue = hsvSrcRowPointer[srcColIndex];
      pixel.saturation = hsvSrcRowPointer[srcColIndex + 1];
      pixel.value = hsvSrcRowPointer[srcColIndex + 2];

      if ((pixel.green > min_green) && (pixel.green > (pixel.red * 1.15)) &&
          (pixel.green > (pixel.blue * 1.15))) {
        destRowPointer[j] = 255;
      }
    }
  }

  return newMatrix;
}

struct ReadFrameResult {
  Action action;
  cv::Point center_of_mass;
};

// Reads the current camera frame, loads it into mutable_frame,
// and displays the camera frame in a separate window.
ReadFrameResult read_frame(VideoCapture &camera, Mat &mutable_frame) {
  camera.read(mutable_frame);
  Mat green_filter = apply_green_filter(mutable_frame, /*min_green=*/100);
  int max_area;
  Point max_center_of_mass;
  Mat contours =
      draw_contours(green_filter, max_area, max_center_of_mass, 10000);

  Scalar color = Scalar(255, 0, 0); // Color for Drawing tool

  Mat frame_gray;
  cv::cvtColor(mutable_frame, frame_gray, cv::COLOR_BGR2GRAY);

  cv::circle(mutable_frame, max_center_of_mass, 20, color, 10);
  imshow("", mutable_frame);
  // imshow("Gray", frame_gray);

  ReadFrameResult res;
  res.center_of_mass = max_center_of_mass;

  if (max_center_of_mass.x == 0) {
    res.action = NONE;
    return res;
  }

  if (max_center_of_mass.x < 1000) {
    res.action = MOVE_RIGHT;
  } else if (max_center_of_mass.x > 1200) {
    res.action = MOVE_LEFT;
  } else {
    res.action = NONE;
  }
  return res;
}

int clamp(int value, int low, int high) {
  if (value < low) {
    return low;
  } else if (value > high) {
    return high;
  }
  return value;
}

void main_loop() {
  Mat frame;
  // Initialize camera
  VideoCapture cap;
  cap.open(0);
  if (!cap.isOpened()) {
    std::cout << "Unable to open camera" << std::endl;
    return;
  }

  while (1) {
    ReadFrameResult frameinput = read_frame(cap, frame);
    cv::waitKey(2);
    std::this_thread::yield();
  }
}

void input_thread() {
    std::cout<< "Launching input thread";
    while (1) {
        std::this_thread::sleep_for (std::chrono::seconds(1));
        std::cout<<"thread id" << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::thread input_loop(input_thread);
    main_loop();
    input_loop.join();
  return 0;
}
