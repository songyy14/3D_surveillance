from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

libdr = ['C:/D/opencv/build/x64/vc14/lib']
incdr = [numpy.get_include(), 'C:/D/opencv/build/include']

ext = [
	Extension('cvt', ['python/cvt.pyx'], 
		language = 'c++', 
		extra_compile_args = ['-std=c++11'], 
		include_dirs = incdr, 
		library_dirs = libdr, 
		libraries = ['opencv_world310']), 
	Extension('KCF', ['python/KCF.pyx', 'C:/E/C++/Tracking/KCF_test/kcftracker.cpp', 'C:/E/C++/Tracking/KCF_test/fhog.cpp'], 
		language = 'c++', 
		extra_compile_args = ['-std=c++11'], 
		include_dirs = incdr, 
		library_dirs = libdr, 
		libraries = ['opencv_world310'])
]

setup(
	name = 'app', 
	cmdclass = {'build_ext':build_ext}, 
	ext_modules = ext
)

#python setup.py build_ext --inplace
