root=$(pwd)
extensions=$(realpath extensions)

# jsrelu C++ extensions
cd $root/codes/modules/jsrelu_ext/
python setup.py install --prefix $extensions