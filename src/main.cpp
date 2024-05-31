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

py::array run_seg(py::array_t<uint8_t>& img, py::array_t<int> hues, py::array_t<int> colors, int frametype = Utils::FrameType::Tracked) {
  Utils::HUE hueList = {};
  for (int i = 0; i < hues.size(); i++) {
    hueList.push_back({(double) hues.at(i), colors.at(i)});
  }

  Vision& vis = Vision::singleton(hueList);

  py::buffer_info buf = img.request();
  cv::Mat frame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

  cv::Mat image = vis.update(frame, (Utils::FrameType) frametype);

  return py::array({image.rows, image.cols, static_cast<int>(image.channels())}, image.data);
}


PYBIND11_MODULE(vss_vision, m) {
  m.doc() = "vss-vision lib";
  m.def("run_seg", run_seg, "run segmentation on frame");
}
