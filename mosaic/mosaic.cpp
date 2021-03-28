#include "mosaic.h"

#include <utility>

MosaicGenerator::MosaicGenerator(int color_depth_, std::vector<double> scales_, std::vector<int> shift_)
    : color_depth(color_depth_), scales(std::move(scales_)), shift(std::move(shift_)) {}
void MosaicGenerator::ColorQuantization() {
  Mat_<Vec3b>::iterator iterator_start;
  iterator_start = input_image.begin<Vec3b>();
  Mat_<Vec3b>::iterator iterator_end;
  iterator_end = input_image.end<Vec3b>();
  for (; iterator_start != iterator_end; iterator_start++) {
    (*iterator_start)[0] = (*iterator_start)[0] / color_depth * color_depth + color_depth / 2;
    (*iterator_start)[1] = (*iterator_start)[1] / color_depth * color_depth + color_depth / 2;
    (*iterator_start)[2] = (*iterator_start)[2] / color_depth * color_depth + color_depth / 2;
  }
}
cv::Mat MosaicGenerator::GenerateMosaic(const string &tile_path, const string &image_path) {
  cv::Mat tile = ReadImage(tile_path);
  tiles = tile_generator.GenerateTiles(tile, scales);
  input_image = ReadImage(image_path);
  ColorQuantization();
  auto boxes = ProcessImageBoxes();
  CreateTiledMosaic(boxes);
  return output_image;
}
cv::Mat MosaicGenerator::ReadImage(const string &image_path) {
  cv::Mat image = imread(image_path, CV_8UC4);
  if (image.empty()) {
    throw std::invalid_argument("Wrong path");
  }
  return image;
}
MosaicGenerator::MosaicGenerator(int color_depth_,
                                 std::vector<double> scales_,
                                 std::vector<int> shift_,
                                 const TileGenerator &tile_generator_)
    : tile_generator(tile_generator_),
      color_depth(color_depth_),
      scales(std::move(scales_)),
      shift(std::move(shift_)) {}
cv::Scalar FindMostFrequentColor(const Mat &image) {
  cv::Mat data;
  image.convertTo(data, CV_32F);
  data = data.reshape(1, data.total());
  cv::Mat labels, centers;
  kmeans(data, 1, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 1.0), 3,
         KMEANS_PP_CENTERS, centers);
  centers = centers.reshape(4, centers.rows);
  cv::Scalar color = centers.at<Vec3f>(0, 0);
  if (color != cv::Scalar::zeros()) {
    return color;
  } else {
    return {0, 0, 0, 0};
  }
}
std::vector<Box> MosaicGenerator::ImageSplitBox(const Mat &image, const cv::Size &resolution) {
  std::vector<int> box_shift;
  if (shift.empty()) {
    box_shift = {resolution.width, resolution.height};
  } else {
    box_shift = shift;
  }
  std::vector<Box> boxes;
  for (int y = 0; y < image.size[1]; y += box_shift[1]) {
    for (int x = 0; x < image.size[0]; x += box_shift[0]) {
      cv::Rect roi = cv::Rect(cv::Point(x, y), cv::Point(x + resolution.width, y + resolution.height));
      cv::Rect clipped_roi = roi & cv::Rect(0, 0, image.size().width, image.size().height);
      if (roi == clipped_roi) {
        cv::Mat boxed_image = image(roi);
        auto mfc = FindMostFrequentColor(boxed_image);
        boxes.emplace_back(Box(boxed_image, cv::Point(x, y), mfc));
      }
    }
  }
  return boxes;
}
std::pair<Tile, double> MosaicGenerator::MostSimilarTile(const cv::Scalar &box_color, const std::vector<Tile> &tiles_) {
  Tile result = tiles_[0];
  double result_distance = 0;
  double minimal_distance = 1000000;
  for (const auto &tile : tiles_) {
    double distance = cv::norm(box_color, tile.color, NORM_L2);
    if (distance < minimal_distance) {
      result = tile;
      result_distance = distance;
      minimal_distance = distance;
    }
  }
  return std::make_pair(result, result_distance);
}
std::vector<Box> MosaicGenerator::ProcessImageBoxes() {
  std::vector<Box> all_boxes;
  for (const auto &tile : tiles) {
    std::vector<Box> boxes = ImageSplitBox(input_image, tile.resolution);
    std::vector<std::pair<Tile, double>> most_similar_tiles;
    for (const auto &box : boxes) {
      cv::Scalar box_color = box.color;
      auto most_similar_pair = MostSimilarTile(box_color, tiles);
      most_similar_tiles.emplace_back(most_similar_pair);
    }
    int i = 0;
    for (const auto &mst : most_similar_tiles) {
      boxes[i].tile = mst.first;
      boxes[i].distance = mst.second;
      i++;
    }
    all_boxes.insert(all_boxes.end(), boxes.begin(), boxes.end());
  }

  return all_boxes;
}
void MosaicGenerator::TilePlacement(Box box) {
  cv::Point p1 = box.position;
  cv::Point p2 = p1 + cv::Point(box.image.size[0], box.image.size[1]);
  cv::Mat image_box = output_image(cv::Rect(p1, p2));
  cv::Mat mask, alpha_plane;
  std::vector<cv::Mat> planes;
  cv::split(box.tile.image, planes);
  alpha_plane = planes[3];
  cv::threshold(alpha_plane, mask, 0, 255, 0);
  mask = mask(cv::Rect(cv::Point(0, 0), cv::Point(image_box.size[0], image_box.size[1])));
  cv::Mat mask_diff;
  cv::bitwise_and(image_box, image_box, mask_diff, mask);
  if (fabs(cv::sum(mask_diff)[0]) < std::numeric_limits<double>::min()) {
    box.tile.image(cv::Rect(cv::Point(0, 0),
                            cv::Point(image_box.size[0], image_box.size[1]))).copyTo(output_image(cv::Rect(p1, p2)));
  }
}
void MosaicGenerator::CreateTiledMosaic(std::vector<Box> boxes) {
  output_image = cv::Mat(input_image.size(), CV_8UC4, {0, 0, 0, 0});
  std::sort(boxes.begin(), boxes.end());
  for (const auto &box : boxes) {
    TilePlacement(box);
  }
}

