#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Entity/Entity.h"
#include "Field/Field.h"
#include "Vision/PositionProcessing/runlengthencoding.h"
#include "Vision/Vision.h"
#include "Utils/Utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>

namespace py=pybind11;

py::array run_detect(py::array_t<uint8_t>& img, py::array_t<int> hues, py::array_t<int> colors) {
  Utils::HUE hueList = {};
  hueList.push_back({(double) hues.at(0), -1});
  for (int i = 0; i < hues.size()-1; i++) {
    hueList.push_back({(double) hues.at(i+1), colors.at(i)});
  }

  Vision& vis = Vision::singleton(hueList);

  py::buffer_info buf = img.request();
  cv::Mat frame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

  PositionProcessing::Blobs blobs = vis.detect(frame);

  // transform to py array
  std::vector<int> shape = {static_cast<int>(blobs.size()), 6};
  py::array_t<int> py_blobs(shape);
  auto py_blobs_unchecked = py_blobs.mutable_unchecked();
  for (int i = 0; i < blobs.size(); i++) {
    py_blobs_unchecked(i, 0) = blobs[i].id;
    py_blobs_unchecked(i, 1) = blobs[i].position.x;
    py_blobs_unchecked(i, 2) = blobs[i].position.y;
    py_blobs_unchecked(i, 3) = blobs[i].angle;
    py_blobs_unchecked(i, 4) = blobs[i].color;
    py_blobs_unchecked(i, 5) = blobs[i].area;
  }

  return py_blobs;
}


py::array run_seg(py::array_t<uint8_t>& img, py::array_t<int> hues, py::array_t<int> colors) {
  Utils::HUE hueList = {};
  hueList.push_back({(double) hues.at(0), -1});
  for (int i = 0; i < hues.size()-1; i++) {
    hueList.push_back({(double) hues.at(i+1), colors.at(i)});
  }

  Vision& vis = Vision::singleton(hueList);

  py::buffer_info buf = img.request();
  cv::Mat frame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

  cv::Mat image = vis.update(frame, Utils::FrameType::Segmented);

  return py::array({image.rows, image.cols, static_cast<int>(image.channels())}, image.data);
}


PYBIND11_MODULE(vss_vision, m) {
  m.doc() = "vss-vision lib";
  m.def("run_seg", run_seg, "run segmentation on frame");
  m.def("run_detect", run_detect, "run blobs detection on frame");
}
