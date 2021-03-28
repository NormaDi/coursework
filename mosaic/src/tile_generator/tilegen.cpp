#include "tilegen.h"

void TileGenerator::CalculateColorVector() {
  for (int bi = 0; bi < depth; bi++) {
    for (int gi = 0; gi < depth; gi++) {
      for (int ri = 0; ri < depth; ri++) {
        auto color = cv::Scalar(int(double(bi) / double(depth) * 255),
                                int(double(gi) / double(depth) * 255),
                                int(double(ri) / double(depth) * 255),
                                255);
        colors.push_back(color);
      }
    }
  }
}

std::vector<Tile> TileGenerator::GenerateTiles(const cv::Mat &image, const std::vector<double> &scales) {
  std::vector<Tile> tiles;
  for (auto scale : scales) {
    for (const auto &color : colors) {
      cv::Mat resized_image;
      cv::resize(image, resized_image, cv::Size(int(image.size[0] * scale), int(image.size[1] * scale)));
      std::vector<Mat> planes;
      cv::split(resized_image, planes);
      cv::Mat mask, alpha_plane = planes[3];
      cv::threshold(alpha_plane, mask, 0, 255, 0);
      cv::Mat colored_image;
      resized_image.copyTo(colored_image);
      colored_image.setTo(color, mask);
      tiles.emplace_back(colored_image, color);
    }
  }
  return tiles;
}