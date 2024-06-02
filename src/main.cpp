#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "Vision/PositionProcessing/PositionProcessing.h"
#include "Vision/Vision.h"
#include "Utils/Utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>

namespace py=pybind11;

using Blob = PositionProcessing::Blob;
using Region = PositionProcessing::Region;
using FieldRegions = PositionProcessing::FieldRegions;
using BlobsEntities = PositionProcessing::BlobsEntities;


py::dict convertEntitiesToPyDict(const BlobsEntities& entities) {
    py::dict py_entitites;

    auto convert_regions = [](const std::vector<Region>& regions) {
        py::list py_regions;
        for (const auto& region : regions) {
            py::dict py_region;
            py::list py_blobs;
            for (const auto& blob : region.blobs) {
                py::dict py_blob;
                py_blob["id"] = blob.id;
                py_blob["position"] = py::make_tuple(blob.position.x, blob.position.y);
                py_blob["angle"] = blob.angle;
                py_blob["valid"] = blob.valid;
                py_blob["area"] = blob.area;
                py_blob["color"] = blob.color;
                py_blobs.append(py_blob);
            }
            py_region["blobs"] = py_blobs;
            py_region["team"] = region.team;
            py_region["distance"] = region.distance;
            py_regions.append(py_region);
        }
        return py_regions;
    };

    py_entitites["team"] = convert_regions(entities.team);
    py_entitites["enemies"] = convert_regions(entities.enemies);

    py::dict py_ball;

    py_ball["id"] = entities.ball.id;
    py_ball["position"] = py::make_tuple(entities.ball.position.x, entities.ball.position.y);
    py_ball["angle"] = entities.ball.angle;
    py_ball["valid"] = entities.ball.valid;
    py_ball["area"] = entities.ball.area;
    py_ball["color"] = entities.ball.color;

    py_entitites["ball"] = py_ball;

    return py_entitites;
}

py::dict run_detect(py::array_t<uint8_t>& img, py::array_t<int> hues, py::array_t<int> colors) {
  Utils::HUE hueList = {};
  hueList.push_back({(double) hues.at(0), -1});
  for (int i = 0; i < hues.size()-1; i++) {
    hueList.push_back({(double) hues.at(i+1), colors.at(i)});
  }

  Vision& vis = Vision::singleton(hueList);

  py::buffer_info buf = img.request();
  cv::Mat frame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

  BlobsEntities regions = vis.detect(frame);

  return convertEntitiesToPyDict(regions);
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

  py::class_<cv::Point>(m, "Point")
      .def(py::init<>())
      .def(py::init<int, int>())
      .def_readwrite("x", &cv::Point::x)
      .def_readwrite("y", &cv::Point::y);

  py::class_<Blob>(m, "Blob")
      .def(py::init<>())
      .def_readwrite("id", &Blob::id)
      .def_readwrite("position", &Blob::position)
      .def_readwrite("angle", &Blob::angle)
      .def_readwrite("valid", &Blob::valid)
      .def_readwrite("area", &Blob::area)
      .def_readwrite("color", &Blob::color);

  py::class_<Region>(m, "Region")
      .def(py::init<>())
      .def_readwrite("blobs", &Region::blobs)
      .def_readwrite("team", &Region::team)
      .def_readwrite("distance", &Region::distance);

  py::class_<FieldRegions>(m, "FieldRegions")
      .def(py::init<>())
      .def_readwrite("team", &FieldRegions::team)
      .def_readwrite("enemies", &FieldRegions::enemies);
}
