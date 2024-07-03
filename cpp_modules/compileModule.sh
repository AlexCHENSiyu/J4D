# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 OptD_module.cpp -o ../SWEmatching/OptD.so
# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 HDQ_module.cpp -o ../HDQ.so
# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SDQ_module.cpp -o ../SWEmatching/SDQ.so
# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 fixed_delta_C_module.cpp -o ../fixed_delta_C.so

g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 rate_calc.cpp -o ../rate_calc.so

# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SWE_OptS_QF_fixed_module.cpp -o ../SWEmatching/SWE_OptS_QF_fixed.so
# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SWE_OptD_QF_fixed_module.cpp -o ../SWEmatching/SWE_OptD_QF_fixed.so
# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SWE_JPEG_d_fixed_module.cpp -o ../SWEmatching/SWE_JPEG_d_fixed.so
# g++ -shared -fPIC -std=c++11 -I./pybind11/include/ -I/usr/include/python3.8 SWE_OptD_d_fixed_module.cpp -o ../SWEmatching/SWE_OptD_d_fixed.so
