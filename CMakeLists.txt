cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0023 NEW)

project(myops)

#find_package(PythonInterp)
find_package(PythonLibs)
#find_package(PythonExtensions)

add_library(myops SHARED
	c_source.cpp)
target_include_directories(myops PRIVATE ${PYTORCH_INCLUDES} ${PYTHON_INCLUDE_DIRS})
set_property(TARGET myops PROPERTY CXX_STANDARD 14)

get_property(_TEMP TARGET myops PROPERTY LINK_FLAGS)
set_property(TARGET myops PROPERTY LINK_FLAGS "${_TEMP} ${PYTORCH_LDFLAGS}")

install(TARGETS myops
        LIBRARY DESTINATION pytorch_register_op_minimal
        RUNTIME DESTINATION pytorch_register_op_minimal)
