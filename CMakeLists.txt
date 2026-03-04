cmake_minimum_required(VERSION 3.20)

project(onnxruntime_ep_telum LANGUAGES C CXX)

# This repo builds a standalone ONNX Runtime plugin Execution Provider (EP) shared library.
# ORT loads the library at runtime via Ort::Env::RegisterExecutionProviderLibrary.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TELUM_EP_ENABLE_ZDNN "Enable optional zDNN backend (Linux s390x only)" ON)

# Path to ORT public headers. Must contain:
#   onnxruntime_cxx_api.h
#   onnxruntime_ep_c_api.h
set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "Path to onnxruntime/core/session public headers")

if(NOT ONNXRUNTIME_INCLUDE_DIR)
  message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR is required. Point it at ORT's include/onnxruntime/core/session.")
endif()
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR does not contain onnxruntime_cxx_api.h: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_ep_c_api.h")
  message(FATAL_ERROR "ONNXRUNTIME_INCLUDE_DIR does not contain onnxruntime_ep_c_api.h: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

# Header-only dependency used by the ORT sample plugin EP utilities.
# We keep this flexible:
# - If GSL_INCLUDE_DIR is set, use it.
# - Otherwise fetch Microsoft.GSL as a build-time dependency.
set(GSL_INCLUDE_DIR "" CACHE PATH "Path to Microsoft.GSL include dir (must contain gsl/span)")

if(NOT GSL_INCLUDE_DIR)
  include(FetchContent)
  FetchContent_Declare(
    microsoft_gsl
    GIT_REPOSITORY https://github.com/microsoft/GSL.git
    GIT_TAG v4.1.0
  )
  FetchContent_MakeAvailable(microsoft_gsl)
  set(GSL_INCLUDE_DIR "${microsoft_gsl_SOURCE_DIR}/include")
endif()

if(NOT EXISTS "${GSL_INCLUDE_DIR}/gsl/span")
  message(FATAL_ERROR "GSL_INCLUDE_DIR does not contain gsl/span: ${GSL_INCLUDE_DIR}")
endif()

file(GLOB_RECURSE TELUM_PLUGIN_EP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/telum_plugin_ep/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/telum_plugin_ep/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin_ep_utils.h"
)

add_library(telum_plugin_ep SHARED ${TELUM_PLUGIN_EP_SOURCES})
target_include_directories(telum_plugin_ep PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${GSL_INCLUDE_DIR}"
)

# Optional zDNN backend path (runtime dlopen/dlsym; no compile-time zDNN dependency).
if(TELUM_EP_ENABLE_ZDNN AND UNIX AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x|s390)$")
  target_compile_definitions(telum_plugin_ep PRIVATE ORT_TELUM_PLUGIN_EP_ZDNN=1)
  if(CMAKE_DL_LIBS)
    target_link_libraries(telum_plugin_ep PRIVATE ${CMAKE_DL_LIBS})
  endif()
endif()

# Constrain exported symbols to only the plugin EP entrypoints (CreateEpFactories/ReleaseEpFactory).
if(UNIX)
  if(APPLE)
    target_link_options(telum_plugin_ep PRIVATE "SHELL:-Wl,-dead_strip")
  elseif(NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
    target_link_options(telum_plugin_ep PRIVATE
      "SHELL:-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/telum_plugin_ep/telum_plugin_ep.lds"
      "SHELL:-Wl,--no-undefined"
      "SHELL:-Wl,--gc-sections"
      "SHELL:-Wl,-z,noexecstack"
    )
  endif()
else()
  # Windows/MSVC: export list via .def file.
  set_property(TARGET telum_plugin_ep APPEND_STRING PROPERTY LINK_FLAGS
    " -DEF:${CMAKE_CURRENT_SOURCE_DIR}/src/telum_plugin_ep/telum_plugin_ep.def"
  )
endif()
