#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <utility>

#pragma once

using namespace cv;
using namespace std;

//! Структура для представления плитки
struct Tile {
  //! Изображение плитки
  cv::Mat image{};
  //! Цвет плитки
  cv::Scalar color{};
  //! Размер плитки
  cv::Size resolution{};
  Tile(Mat image_, cv::Scalar color_) : image(std::move(image_)), color(std::move(color_)) {
    resolution = image.size();
  };
  Tile() = default;
};

//! Класс для создания плиток из изображения
class TileGenerator {
 public:
  //! Стандартный конструктор (для создания пустого объекта генератора).
  TileGenerator() = default;
  //! Деструктор.
  ~TileGenerator() = default;
  //! Конструктор копирования.
  TileGenerator(const TileGenerator &rhs) = default;
  //! Конструктор.
  explicit TileGenerator(int depth_) : depth(depth_) { CalculateColorVector(); };
  //! Метод для генерации плиток.
  std::vector<Tile> GenerateTiles(const cv::Mat &image, const std::vector<double> &scales);
 private:
  //! Метод для создания вектора цвета.
  void CalculateColorVector();

  //! Глубина изображения плитки.
  int depth{};

  //! Цвета (с учётом глубины)
  std::vector<cv::Scalar> colors;
};


