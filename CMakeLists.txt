cmake_minimum_required(VERSION 3.6)
project(facetracker)

set(CMAKE_CXX_STANDARD 14)

file(COPY shape_predictor_68_face_landmarks.dat DESTINATION ${PROJECT_BINARY_DIR}/lib)

# OpenBLAS
include_directories(/usr/local/Cellar/openblas/0.2.20/include)
link_directories(/usr/local/Cellar/openblas/0.2.20/lib)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

AUX_SOURCE_DIRECTORY(mtcnn MTCNN_SRCS)
set(SOURCE_FILES main.cpp ${MTCNN_SRCS})
add_executable(tracker ${SOURCE_FILES})

target_link_libraries(tracker ${OpenCV_LIBS})
target_link_libraries(tracker openblas)

find_package(dlib)
target_link_libraries(tracker dlib::dlib)