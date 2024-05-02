#include <CameraManager/CameraThread.h>
#include <TBBThreadManager.h>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <qelapsedtimer.h>
#include <visionthread.h>
#include <QApplication>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "CameraManager/CameraManager.h"
#include "Entity/Entity.h"
#include "Field/Field.h"
#include "Vision/PositionProcessing/runlengthencoding.h"
#include "Vision/Vision.h"
#include "Windows/MainVSSWindow.h"
#include "maggicsegmentationdialog.h"

int main(int argc, char *argv[]) {
  cv::useOptimized();
  Logging::init();

  Vision& vis = Vision::singleton();

  // cv::Mat frame = cv::Mat::zeros(FRAME_HEIGHT_DEFAULT, FRAME_WIDTH_DEFAULT, CV_8UC3) * 255;
  cv::Mat frame = cv::imread("../image.png");

  cv::Mat res = vis.update(frame, QTime::currentTime());

  std::cout << "Processing done" << std::endl;
  cv::imwrite("../segmented.png", res);

  return 0;
}
