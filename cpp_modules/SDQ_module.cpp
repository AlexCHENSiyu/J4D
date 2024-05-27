// SDQ_module.cpp

// MIT License

// Copyright (c) 2022 deponce(Linfeng Ye), University of Waterloo

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./SDQ/SDQ.h"
using namespace std;
namespace py = pybind11;
// TODO: REMOVE string Model, int colorspace; ADD ColorSpaceW InvColorSpaceW BiasPerImage
std::pair<py::array, float> py__call__(py::array_t<float, py::array::c_style | py::array::forcecast> array,
                     py::array_t<float, py::array::c_style | py::array::forcecast> SenMap,
                     py::array_t<float, py::array::c_style | py::array::forcecast> ColorSpaceW_,
                     py::array_t<float, py::array::c_style | py::array::forcecast> InvColorSpaceW_,
                     bool BiasPerImageFlag,
                     int J, int a, int b, 
                     py::array_t<float> QT_Y_,
                     py::array_t<float> QT_C_,
                     float Beta_S, float Beta_W, float Beta_X, 
                     float Lmbd, float eps, int iterations){
  /*
  :Fn  py__call__: 
  :param py::array_t<float, py::array::c_style | py::array::forcecast> array: 
  :param int J, int a, int b:
  :param py::array_t<float> QT_Y_:
  :param py::array_t<float> QT_C_:
  :param float Beta_S:
  :param float Beta_W:
  :param float Beta_X:
  :param float Lmbd:
  :param float eps: 
  :return: std::pair<py::array, float>
  */
  unsigned long size[2];
  size[0] = (unsigned long)array.shape()[1];
  size[1] = (unsigned long)array.shape()[2];
  // allocate std::vector (to pass to the C++ function)
  vector<float> pos(array.size());
  vector<vector<vector<float>>> Vect_img(3, vector<vector<float>>(size[0], vector<float>(size[1], 0)));
  // copy py::array -> std::vector
  float BPP =0;
  vector<float> result(array.size());
  memcpy(pos.data(), array.data(),array.size()*sizeof(float));
  
  float ColorSpaceW[3][3];
  float InvColorSpaceW[3][3];

  memcpy(ColorSpaceW, ColorSpaceW_.data(), 3*3*sizeof(float));
  memcpy(InvColorSpaceW, InvColorSpaceW_.data(),3*3*sizeof(float));
  
  float QT_Y[64];
  float QT_C[64];

  memcpy(QT_Y, QT_Y_.data(), 64*sizeof(float));
  memcpy(QT_C, QT_C_.data(), 64*sizeof(float));
  // std::cout<<"a"<<std::flush<<std::endl;
  // call pure C++ function
  seq2img(pos, Vect_img, size[0], size[1]);
  float Sen_Map[3][64]={0};
  memcpy(Sen_Map, SenMap.data(), 3*64*sizeof(float));

  float W_rgb2swx[3][3];
  float W_swx2rgb[3][3];
  float bias_rgb2swx = 128.;
  float ImageBias[3] = {0,0,0};

  ColorSpaceConv(Vect_img, ColorSpaceW, ImageBias, BiasPerImageFlag);

  SDQ sdq;
  sdq.__init__(eps, iterations, Beta_S, Beta_W, Beta_X, Lmbd, Sen_Map, QT_Y, QT_C, J, a ,b);
  
  BPP = sdq.__call__(Vect_img); // Vect_img is the compressed dequantilzed image after sdq.__call__()
  
  InvColorSpaceConv(Vect_img, InvColorSpaceW, ImageBias);

  img2seq(Vect_img, result, size[0], size[1]);
  int ndim = 3;
  vector<unsigned long> shape   = { 3, size[0], size[1]};
  vector<unsigned long> strides = { size[0]*size[1]*sizeof(float),
                                    size[1]*sizeof(float), sizeof(float)};
                                    
  return std::make_pair(py::array(py::buffer_info(
    result.data(),                           /* data as contiguous array */
    sizeof(float),                           /* size of one scalar       */
    py::format_descriptor<float>::format(),  /* data type                */
    ndim,                                    /* number of dimensions     */
    shape,                                   /* shape of the matrix      */
    strides                                  /* strides for each axis    */
  )), BPP);
}
// wrap as Python module
PYBIND11_MODULE(SDQ,m)
{
  m.doc() = "SDQ API";
  m.def("__call__", &py__call__, py::return_value_policy::move ,"Calculate the length of an array of vectors");
}
