PROJECT(Mosaic_lib)

add_library(${PROJECT_NAME} mosaic.h mosaic.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Tilegen_lib)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)