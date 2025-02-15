#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "Vision/PositionProcessing/PositionProcessing.h"
#include "Vision/Vision.h"
#include "Utils/Utils.h"
#include "Entity/Entity.h"
#include "GameInfo/GameInfo.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include <pybind11/pytypes.h>

namespace py=pybind11;

using Blob = PositionProcessing::Blob;
using Region = PositionProcessing::Region;
using FieldRegions = PositionProcessing::FieldRegions;
using BlobsEntities = PositionProcessing::BlobsEntities;
using GameInfo = GameInfo;


py::dict convertEntitiesToPyDict(const GameInfo& gameInfo) {
    py::dict py_gameInfo;

    auto convert_players = [](const std::vector<Player>& players) {
        py::list py_players;
        for (const Entity& player : players) {
            py::dict py_player;
            py_player["id"] = player.m_id;
            py_player["position"] = py::make_tuple(player.m_position.x, player.m_position.y);
            py_player["angle"] = player.m_angle;
            py_player["team"] = player.m_team;
            py_players.append(py_player());
        }
        return py_players;
    };

    py_gameInfo["players"] = convert_players(gameInfo.m_players);

    py::dict py_ball;
    
    Entity ball = gameInfo.m_ball;
    py_ball["id"] = ball.m_id;
    py_ball["position"] = py::make_tuple(ball.m_position.x, ball.m_position.y);
    py_ball["angle"] = ball.m_angle;

    py_gameInfo["ball"] = py_ball;

    return py_gameInfo;
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

  GameInfo gameInfo = vis.detect(frame);

  return convertEntitiesToPyDict(gameInfo);
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
