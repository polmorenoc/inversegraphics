#!/usr/bin/env python3.4m
 
from distutils.core import setup
from distutils.extension import Extension
 
setup(name="PackageName",
    ext_modules=[
        Extension("hello", ["test.cpp"],
        libraries = ["boost_python"])
    ])