#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// #include <opencv2/highgui.hpp>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./HDQ/DCT_BPP.h"
// #include "./Utils/ColorSpaceConv.h"
// using namespace cv;
using namespace std;

// -------------
// pure C++ code
// -------------
using namespace std;

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
float py__call__(py::array_t<float, py::array::c_style | py::array::forcecast> array)
{
  unsigned long size[2];
  size[0] = (unsigned long)array.shape()[1];
  size[1] = (unsigned long)array.shape()[2];
  // allocate std::vector (to pass to the C++ function)
  vector<float> pos(array.size());
  vector<vector<vector<float>>> Vect_img(3, vector<vector<float>>(size[0], vector<float>(size[1], 0)));
  // copy py::array -> std::vector
  memcpy(pos.data(), array.data(),array.size()*sizeof(float));

  // call pure C++ function
  //TODO::
  float BPP = 0;
  vector<float> result(array.size());
  seq2img(pos, Vect_img, size[0], size[1]);
  DCT_BPP dct_bpp;
  dct_bpp.__init__();
  BPP = dct_bpp.__call__(Vect_img); // Vect_img is the compressed dequantilzed image after sdq.__call__()
  return BPP;
}
// wrap as Python module
PYBIND11_MODULE(rate_calc ,m)
{
  m.doc() = "rate_calc API";
  m.def("__call__", &py__call__, py::return_value_policy::move ,"Calculate the length of an array of vectors");
}
