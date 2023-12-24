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
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>

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
void SDL_Fail() {
  SDL_LogError(SDL_LOG_CATEGORY_CUSTOM, "Error %s", SDL_GetError());
  exit(1);
}

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
ReadFrameResult read_frame(VideoCapture &camera, Mat &mutable_frame, cv::CascadeClassifier &body_cascade, cv::CascadeClassifier &face_cascade, cv::CascadeClassifier &eye_cascade) {
  camera.read(mutable_frame);
  Mat green_filter = apply_green_filter(mutable_frame, /*min_green=*/ 100);
  int max_area;
  Point max_center_of_mass;
  Mat contours = draw_contours(green_filter, max_area, max_center_of_mass, 10000);

  Scalar color = Scalar(255, 0, 0); // Color for Drawing tool

  Mat frame_gray;
  cv::cvtColor( mutable_frame, frame_gray, cv::COLOR_BGR2GRAY );

    cv::circle(mutable_frame, max_center_of_mass, 20, color, 10);
  imshow("Live", mutable_frame);
    imshow("Gray", frame_gray);
  
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

void draw_text(SDL_Renderer *screen, char *string, int size, float x, float y,
               Uint8 fR, Uint8 fG, Uint8 fB, Uint8 bR, Uint8 bG, Uint8 bB) {

  TTF_Font *font = TTF_OpenFontDPI("Arial.ttf", 10, 200, 200);
  if (font == NULL) {
    SDL_Log("Error with loading font: %s", SDL_GetError());
  }

  SDL_Color foregroundColor = {fR, fG, fB};
  SDL_Color backgroundColor = {bR, bG, bB};

  SDL_Surface *textSurface =
      TTF_RenderText_Shaded(font, string, foregroundColor, backgroundColor);

  SDL_FRect textLocation = {x, y, 200, 50};

  SDL_Texture *tex = SDL_CreateTextureFromSurface(screen, textSurface);
  if (tex == NULL) {
    SDL_Log("Error with loading texture: %s", SDL_GetError());
  }

  SDL_RenderTexture(screen, tex, NULL, &textLocation);

  SDL_DestroySurface(textSurface);
  SDL_DestroyTexture(tex);

  TTF_CloseFont(font);
}

void draw_hud(cv::Point centerOfMass, Action action) {}

void main_loop(SDL_Window *win) {
  Mat frame;
  // Initialize camera
  VideoCapture cap;
  cap.open(0);
  if (!cap.isOpened()) {
    SDL_Log("ERROR! Unable to open camera\n");
    return;
  }

  cv::CascadeClassifier body_cascade;
  cv::CascadeClassifier face_cascade;
  cv::CascadeClassifier eye_cascade;
  
  if (!body_cascade.load("haarcascade_fullbody.xml")) {
    SDL_Log("Could not load fullbody cascade");
  }
  if (!face_cascade.load("haarcascade_frontalface_alt.xml")) {
    SDL_Log("Could not load face cascade");
  }
  if (!eye_cascade.load("haarcascade_eye_tree_eyeglasses.xml")) {
    SDL_Log("Could not load eye cascade");
  }

  // Setup basic renderer.
  Uint32 render_flags = SDL_RENDERER_ACCELERATED;
  SDL_Renderer *rend = SDL_CreateRenderer(win, NULL, render_flags);

  SDL_RWops *file = SDL_RWFromFile(image_path, "rb");
  SDL_Surface *sprite = SDL_LoadBMP_RW(file, SDL_TRUE);
  if (!sprite) {
    SDL_Log("Failed to load image: %s", SDL_GetError());
  }
  SDL_Texture *tex = SDL_CreateTextureFromSurface(rend, sprite);

  SDL_FRect rect;
  rect.x = 500;
  rect.y = 500;
  rect.w = 100;
  rect.h = 100;

  SDL_Event event;
  while (app_quit == false) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT) {
        app_quit = true;
        break;
      } else if (event.type == SDL_EVENT_KEY_DOWN) {
      }
    }

    ReadFrameResult frameinput = read_frame(cap, frame, body_cascade, face_cascade, eye_cascade);
    Action action = frameinput.action;
    if (action == MOVE_LEFT) {
      rect.x = clamp(rect.x - 10, 10, winWidth-100);
    } else if (action == MOVE_RIGHT) {
      //
      // void draw_text(SDL_Renderer *screen, char *string, int size, float x,
      // float y,
      //       Uint8 fR, Uint8 fG, Uint8 fB, Uint8 bR, Uint8 bG, Uint8 bB) {
      rect.x = clamp(rect.x + 10, 10, winWidth - 100);
    }
    SDL_RenderClear(rend);
    int res = SDL_RenderTexture(rend, tex, NULL, &rect);

    char center_of_mass_string[256];
    snprintf(center_of_mass_string, 256, "Center of mass: %d", frameinput.center_of_mass.x);
    draw_text(rend, center_of_mass_string, 20, 100, 100, 255, 0, 0, 0, 0, 0);

    if (res != 0) {
      SDL_Log("error with rendering: %s", SDL_GetError());
    }
    SDL_RenderPresent(rend);
  }

  SDL_DestroySurface(sprite);

  SDL_DestroyTexture(tex);
  SDL_DestroyRenderer(rend);
}

int main(int argc, char *argv[]) {

  // init the library, here we make a window so we only need the Video
  // capabilities.
  if (SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Fail();
  }

  if (TTF_Init()) {
    SDL_Fail();
  }

  // create a window
  SDL_Window *window =
      SDL_CreateWindow("Window", winWidth, winHeight, SDL_WINDOW_RESIZABLE);
  if (!window) {
    SDL_Fail();
  }

  // print some information about the window
  SDL_ShowWindow(window);
  {
    int width, height, bbwidth, bbheight;
    SDL_GetWindowSize(window, &width, &height);
    SDL_GetWindowSizeInPixels(window, &bbwidth, &bbheight);
    SDL_Log("Window size: %ix%i", width, height);
    SDL_Log("Backbuffer size: %ix%i", bbwidth, bbheight);
    if (width != bbwidth) {
      SDL_Log("This is a highdpi environment.");
    }
  }

  SDL_Log("Application started successfully!");

  while (!app_quit) {
    main_loop(window);
  }

  // cleanup everything at the end
  SDL_DestroyWindow(window);
  SDL_Quit();
  SDL_Log("Application quit successfully!");
  return 0;
}
