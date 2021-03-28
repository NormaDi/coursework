#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <set>

#pragma once

#include "../tile_generator/tilegen.h"

using namespace cv;
using namespace std;

//! Метод для поиска наиболее частого цвета в изображении
cv::Scalar FindMostFrequentColor(const cv::Mat &image);

struct Box {
  Box(Mat image_, Point position_, const Scalar &color_, Tile tile_ = Tile())
      : image(std::move(image_)), position(std::move(position_)), tile(std::move(tile_)) {
    color = FindMostFrequentColor(image);
  };
  cv::Mat image;
  cv::Scalar color;
  cv::Point position;
  Tile tile{};
  double distance{0};

  bool operator<(const Box &rhs) const {
    return (distance < rhs.distance);
  }
};

//! Класс для создания мозаики
class MosaicGenerator {
 public:
  //! Стандартный конструктор (для создания пустого объекта генератора).
  MosaicGenerator() = default;
  //! Деструктор.
  ~MosaicGenerator() = default;
  //! Конструктор копирования.
  MosaicGenerator(const MosaicGenerator &rhs) = default;
  //! Конструктор
  MosaicGenerator(int color_depth_, std::vector<double> scales_, std::vector<int> shift_);
  //! Конструктор
  MosaicGenerator(int color_depth_,
                  std::vector<double> scales_,
                  std::vector<int> shift_,
                  const TileGenerator &tile_generator_);
  //! Метод, создающий мозаику по данной плитке
  cv::Mat GenerateMosaic(const string &tile_path, const string &image_path);
 private:
  //! Метод для чтения изображения.
  static cv::Mat ReadImage(const string &image_path);

  //! Метод для уменьшения количества цветов в изображении
  void ColorQuantization();

  //! Метод для разбиения изображения на участки
  std::vector<Box> ImageSplitBox(const cv::Mat &image, const cv::Size &resolution);

  //! Метод поиска самой подходящей плитки к участку
  static std::pair<Tile, double> MostSimilarTile(const cv::Scalar &box_color, const std::vector<Tile> &tiles_);

  //! Метод постройки участков и нахождения наилучшей плитки для каждого
  std::vector<Box> ProcessImageBoxes();

  //! Наложение плитки на изображение
  void TilePlacement(Box box);

  //! Создание мозаики
  void CreateTiledMosaic(std::vector<Box> boxes);

  TileGenerator tile_generator{4};
  std::vector<Tile> tiles{};

  cv::Mat input_image{};
  cv::Mat output_image{};

  int color_depth{8};
  std::vector<double> scales;
  std::vector<int> shift;
};

