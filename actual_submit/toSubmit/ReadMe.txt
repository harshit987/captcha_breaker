Dependencies
1. cython
2. keras
3. tensorflow
4. sklearn

Troubleshooting:
1. If .so file doesn't work on your system then run the following command to create compatible .so file
We are using python3 in script.sh, if you are using python then change "python3 setup.py build_ext --inplace" to "python setup.py build_ext --inplace"
$ bash script.sh