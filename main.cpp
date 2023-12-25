#include <SDL.h>
#include <algorithm>
#include <iostream>
#if defined(WINAPI_FAMILY) && (WINAPI_FAMILY == WINAPI_FAMILY_APP)
// On UWP, we need to not have SDL_main otherwise we'll get a linker error
#define SDL_MAIN_HANDLED
#endif
#include <SDL3_ttf/SDL_ttf.h>
#include <SDL_main.h>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

std::mutex quit_mutex;
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

// Represents the transformation of an image, taking in an input of type Mat,
// and returning an output of type Mat. The idea is to string these
// transformations together in a pipeline, to play around with different image
// transforms and see them work to get some intuition of what is going on.
class ImageTransform {
public:
  virtual Mat apply_transform(const Mat &src) = 0;
};

// Represents a series of transforms to be applied to an image matrix.
class Pipeline {
public:
  // Apply the series of transforms represented by this pipeline
  // to the source matrix.
  Mat apply_pipeline(const Mat &src);

  // Adds a transform at the end of this pipeline.
  // Does not take ownership of the ImageTransform object,
  // which is expected to outlive the pipeline object to
  // which it is added.
  void add_transform(ImageTransform *transform);

private:
  // For now we'll just model this as a linear series of transforms,
  // and we can consider changing this to DAG if needed.
  std::vector<ImageTransform *> transforms;
};

void Pipeline::add_transform(ImageTransform *transform) {
  transforms.push_back(transform);
}

Mat Pipeline::apply_pipeline(const Mat &src) {
  if (transforms.size() == 0) {
    Mat transformed_matrix = src;
    return transformed_matrix;
  }
  Mat transformed_matrix = transforms[0]->apply_transform(src);
  for (int i = 1; i < transforms.size(); i++) {
    transformed_matrix = transforms[i]->apply_transform(transformed_matrix);
  }
  return transformed_matrix;
}

// Select green pixels from the camera frame.
Mat apply_green_filter(const Mat &source, int min_green) {
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

class GreenTransform : public ImageTransform {
public:
  Mat apply_transform(const Mat &src) override {
    return apply_green_filter(src, 100);
  }
};

struct ReadFrameResult {
  Action action;
  cv::Point center_of_mass;
};

// Reads the current camera frame, and applies the transforms
// specified by the pipeline to it.
void read_frame(VideoCapture &camera, Pipeline &pipeline) {
  Mat mutable_frame;
  camera.read(mutable_frame);
  Mat transformed = pipeline.apply_pipeline(mutable_frame);

  imshow("Original", mutable_frame);
  imshow("Gray", transformed);
  cv::moveWindow("Gray", 500, 0);
}

int clamp(int value, int low, int high) {
  if (value < low) {
    return low;
  } else if (value > high) {
    return high;
  }
  return value;
}

bool should_quit() {
  bool should_quit;
  {
    std::lock_guard<std::mutex> guard(quit_mutex);
    should_quit = app_quit;
  }
  return should_quit;
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

  GreenTransform greenTransform;
  Pipeline pipeline;
  pipeline.add_transform(&greenTransform);

  while (!should_quit()) {
    read_frame(cap, pipeline);
    cv::waitKey(2);
    std::this_thread::yield();
  }
}

void input_thread() {
  while (1) {
    std::string command;
    std::cout << "playground> ";
    std::cin >> command;

    if (command == "exit") {
      std::lock_guard<std::mutex> guard(quit_mutex);
      app_quit = true;
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  std::thread input_loop(input_thread);
  main_loop();
  input_loop.join();
  return 0;
}
