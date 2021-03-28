#include <iostream>

#include "src/mosaic/mosaic.h"

int main() {
  MosaicGenerator mosaic_generator = MosaicGenerator(32, {0.3, 0.2, 0.1}, {});
  std::string tile, image;
  std::cin >> tile >> image;
  cv::Mat result = mosaic_generator.GenerateMosaic(tile, image);
  imshow("result", result);
  waitKey();
  return 0;
}
