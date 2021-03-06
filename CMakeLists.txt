# cmake_minimum_required(VERSION 3.0)

# project(KalmsnFilterTesting)

# find_package(OpenCV REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

# add_executable(kalman kalman.cpp)

# target_link_libraries(kalman ${OpenCV_LIBS})


cmake_minimum_required(VERSION 3.0)

project(KalmanFilterTesting)

set(EXTRA_OPENCV_PATH "/opt/opencv")
find_package(OpenCV "4.5.0" EXACT REQUIRED PATHS ${EXTRA_OPENCV_PATH})

include_directories(${OpenCV_INCLUDE_DIRS} 
                    ${PROJECT_SOURCE_DIR}
                    modules/csv/src)

add_executable(kalman kalman.cpp)                    
add_executable(kalmanMouse kalmanMouse.cpp)
add_executable(extendedKalmanMouse extendedKalmanMouse.cpp)
add_executable(unscentedKalman unscentedKalman.cpp)

target_link_libraries(kalman ${OpenCV_LIBS})
target_link_libraries(kalmanMouse ${OpenCV_LIBS})
target_link_libraries(extendedKalmanMouse ${OpenCV_LIBS})
target_link_libraries(unscentedKalman ${OpenCV_LIBS})

