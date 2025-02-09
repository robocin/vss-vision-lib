#ifndef MAGGICSEGMENTATION_H
#define MAGGICSEGMENTATION_H

#include "ImageProcessing.h"
#include "Vision/ColorSpace.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/dist_sink.h"

#include <vector>
#include <set>
#include <string>
#include <map>
#include <fstream>
typedef unsigned char uchar;

#define LUTSIZE 16777216

#define LUT_SIZE 16777216*3

#define YMAXLABEL "YMAX"
#define UMAXLABEL "UMAX"
#define VMAXLABEL "VMAX"

#define YMINLABEL "YMIN"
#define UMINLABEL "UMIN"
#define VMINLABEL "VMIN"

#define NOCOLLABEL "NoCOL"
#define ORANGELABEL "Orange"
#define BLUELABEL "Blue"
#define YELLOWLABEL "Yellow"
#define REDLABEL "Red"
#define GREENLABEL "Green"
#define PINKLABEL "Pink"
#define LIGHTBLUELABEL "LightBlue"
#define PURPLELABEL "Purple"
#define BROWNLABEL "Brown"
#define COLORSTRANGELABEL "ColorStrange"

template<typename T>
T max_element_of(T* data, size_t size) {
  T r = data[0];
  for (size_t i = 1; i < size; ++i) {
    r = (data[i] > r ? data[i] : r);
  }
  return r;
}


enum MaggicVisionDebugSelection {
  MaggicVisionDebugSelection_Undefined = 0,
  MaggicVisionDebugSelection_Thresholded,
  MaggicVisionDebugSelection_ExtremeSaturation,
  MaggicVisionDebugSelection_MultipliedResults,
  MaggicVisionDebugSelection_SegmentationFrame,
  MaggicVisionDebugSelection_DetailsFrame
};



typedef std::vector<cv::Rect> Rectangles;

/**
 * @brief    Class for segmentation using a Look Up Table (LUT).
 */
class MaggicSegmentation : public ImageProcessing
{

// #define min(a,b) (a<b?a:b)
// #define max(a,b) (a>b?a:b)

public:
    enum NormalizationMethod {
        NO_NORMALIZATION= 0,
        CHROMATIC_NORMALIZATION,
        VECTOR_NORMALIZATION,
        WEIGHTED_NORMALIZATION,
        NORMALIZATION_METHODS_LENGTH
    };

  bool paused, enableEstimateRobots;
  bool useLoadedValues = false;
  bool m_updateDetails = true;
  bool m_updateFrame = true;
  
  /**
   * @brief    Default Constructor
   */
  MaggicSegmentation(Utils::HUE list);

  /**
   * @brief    Destroys the object.
   */
  ~MaggicSegmentation();

  /**
   * @brief    Apply the algorithm on the given frame
   *
   * @param[in]  frame  The frame to be processed in BGR
   *
   * @return   A one channel's Mat with the labels on the pixels' value
   */
  cv::Mat run(cv::Mat& frame) final;

  /**
   * @brief    Gets the debug frame, a Mat in BGR channel
   *
   * @return   The debug frame.
   */
  void getDebugFrame(cv::Mat& frame) final;

  /**
   * @brief    Gets the segmentation frame from lut. Frame for debug).
   *
   * @return   The segmentation frame from lut, it's use in Vision to pass for other class.
   */
  void getSegmentationFrameFromLUT(cv::Mat& frame);


  static uint BGR2RGBHash(cv::Vec3b &v);

  static uint RGB2RGBHash(cv::Vec3b &v);

  static cv::Vec3b RGBHash2BGR(uint);

  static cv::Vec3b RGBHash2RGB(uint);

  static String RGBHash2String(uint);

  void calibrate(cv::Mat &frame);

  int getFilterGrayThresholdValue();

  void setFilterGrayThresholdValue(int newFilterGrayThreshold);

  void getFilterGrayThresholdValues(int &minimum, int &maximum);

  void setFilterGrayThresholdValues(int minimum, int maximum);

  void setManyTimes(int manyTimes = 1);

  void setEntitiesCount(int entitiesCount = 1);

  int getEntitiesCount();

  void setDebugSelection(MaggicVisionDebugSelection selection);

  void setLearningThresholdValue(bool enabled);

  void getLearningThresholdValue(bool &enabled);

  bool isLearning();

  void setLearningThresholdFrames(uint frames);

  void getLearningThresholdFrames(uint &frames);

  void getCalibrationFrames(uint &frames);

  void updateFilterGrayThresholdValue();

  void setHUETable(bool fromFile = false);

  void generateLUTFromHUE();

  uchar* getLUT();

  void loadHueList(Utils::HUE list);

  void setVectorscopeEnabled(bool enabled);

  void setFilterEnabled(bool enabled);

  void setNormalizedEnabled(bool enabled);

  bool getNormalizedEnabled();

  void setNormalizationMethod(MaggicSegmentation::NormalizationMethod method);

  void getNormalizationMethod(MaggicSegmentation::NormalizationMethod &method);

  MaggicSegmentation::NormalizationMethod getNormalizationMethod();

  void saveSelectedDebug();

  void lock();

  void unlock();

  void updateDetails();

  void updateFrame();

  void setLUTCacheEnable(bool enabled = true);
  bool getLUTCacheEnable();

private:
  int _minimumGrayThreshold = 10, _maximumGrayThreshold = 40, _intervalGrayThreshold = 30;
  static constexpr float div255 = 1.0f / 255.0f;
  static constexpr float div3255 = 1.0f/(255.0f+ 255.0f+ 255.0f);
  bool normalized_color;
  NormalizationMethod normalization_method, selected_normalization_method;
  static const NormalizationMethod default_normalization_method;
  bool _LUTCacheEnable;

  uint _calibrationFrames, _learningThresholdFrames;

  uint _thresholdFrequency[256];
  bool _learningThresholdValue;

  int greatestSumOfAreas = 0;
  int filterGrayThreshold = 30;
  MaggicVisionDebugSelection _debugSelection;

  typedef std::vector<float> ColorDescriptor;
  ColorDescriptor colorDescriptors;
  std::vector<cv::Point2i> componentsCentroids;

  // @TODO : vector with the rectangles of estimated robots
  Rectangles componentsRectangles;

  struct RobotDescriptor {
    ColorDescriptor colors;
  };

  RobotDescriptor robotsDescriptors[8];

  int* _HUETable;
  bool isLUTReady;

  int _manyTimes, _entitiesCount;
  int component_id = 1;
  int n_components = 0;

  bool updateColors = false;

  void filterGray(cv::Mat &d, cv::Mat &o);

  void applyLUT(cv::Mat &input, cv::Mat &output, uchar* LUT);

  inline void filterGray(cv::Vec3b &color, cv::Vec3b &coloro);

  void filterBinarizeColored(cv::Mat &d, cv::Mat &o);

  void filterExtremeSaturation(cv::Mat &d, cv::Mat &o);

  void updateHistogramDescriptors();

  void filterGrain(ColorDescriptor &dest, ColorDescriptor &orig);

  void _layeredAdd(cv::Mat &out, cv::Mat imgA, cv::Mat imgB);

  bool estimateRobots(cv::Mat img, int manyTimes, int n_components_reference = 7);

  void doDetails();

  void removeTopRectangle(Rectangles& rects, cv::Point& p);

  /**
   * @brief    Init the Look Up Table with the already loaded parameters
   */
  void initLUT();

  // vector of pair: <float hueValue, uchar colorLabel>
  Utils::HUE hueList;
  Utils::HUE defaultHueList = {
    {10, Color::RED}, 
    {15.0, Color::ORANGE}, 
    {40.0, Color::YELLOW}, 
    {97.0, Color::GREEN}, 
    {150.0, Color::CYAN}, 
    {200.0, Color::BLUE}, 
    {220.0, Color::PINK}
  };


  cv::Mat colorPalette,
      colorPaletteYUV,
      colorPaletteHSV,
      histo;

  uchar* _LUT;
  uchar** _LUT_CACHE;
  ColorInterval* _calibrationParameters;
  cv::Mat _imageBuffer;
  cv::Mat _imageBufferFiltered;
  cv::Mat _imageBufferHSV;
  cv::Mat _debugFrame;
  cv::Mat _segmentationFrame;
  cv::Mat _extremeSaturation, _multipliedResults, _firstThreshold, _secondThreshold;
  cv::Mat _detailsFrame, _colorDetailsFrame;
  cv::Mat _filterMask;
  cv::Point2i cursorPos;
  std::vector<cv::Point2i> lastCursorPos;
  Rectangles filterRectangles;
  std::mutex mut;
  int pressedId = 0, releasedId = 0;
  int dragpivotId = -1;
  bool pressedLeft = false, pressedRight = false;
  bool releasedLeft = false, releasedRight = false;
  bool colorDistribution = false;
  bool enableFilter = false;

  
  inline int value(const RGB& px) {
    static cv::Mat LUT_BGR2HSV = cv::Mat::zeros(1,LUT_SIZE/3,CV_8UC3);
    static bool preprocessed = false;
    if (!preprocessed) {

      for (int i=0;i<LUT_SIZE/3;i++) {
        uchar r = static_cast<uchar>((i&0x00ff0000) >> 16);
        uchar g = static_cast<uchar>((i&0x0000ff00) >> 8);
        uchar b = static_cast<uchar>(i&0x000000ff);
        LUT_BGR2HSV.at<cv::Vec3b>(0,i) = cv::Vec3b(b,g,r);
      }

      cv::cvtColor(LUT_BGR2HSV,LUT_BGR2HSV,cv::COLOR_BGR2HSV_FULL);

      // std::cout << "PREPROCESSED" << std::endl;

      preprocessed = true;
    }

    uchar r = px.red;
    uchar g = px.green;
    uchar b = px.blue;

    int i = (int(px.red)<<16) + (int(px.green)<<8) + px.blue;

    cv::Vec3b &coloro = LUT_BGR2HSV.at<cv::Vec3b>(0,i);

    bool filter = false;
    if (!this->_filterMask.empty())  {
      // saturation 1 > value 2
      if (coloro[1] > coloro[2]) {
        // value 2, hue 0
        int ii = coloro[2]/2;
        int jj = coloro[0];

        filter = static_cast<bool>(this->_filterMask.at<uchar>(ii,jj));

      } else {
        // saturation 1, hue 0
        int ii = 127 + (255-coloro[1])/2;
        int jj = coloro[0];

        filter = static_cast<bool>(this->_filterMask.at<uchar>(ii,jj));

      }
      if (filter) {
        return 0;
      }
    }

    cv::Vec3b bgr(b,g,r);
    filterGray(bgr,bgr);
    if (bgr[0] == 0 && bgr[1] == 0 && bgr[2] == 0) {
      return 0;
    }
    else {
      return static_cast<uchar>(this->_HUETable[coloro[0]]);
    }
  }

  std::string _colorLabels[NUMBEROFCOLOR] = { NOCOLLABEL,
                        ORANGELABEL,
                        BLUELABEL,
                        YELLOWLABEL,
                        REDLABEL,
                        GREENLABEL,
                        PINKLABEL,
                        LIGHTBLUELABEL,
                        PURPLELABEL,
                        BROWNLABEL,
                        COLORSTRANGELABEL };

};

#endif
