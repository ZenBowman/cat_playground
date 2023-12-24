mkdir build
cd build
cmake ../
cmake --build ../
cd ..

mkdir xcodeproject
cd xcodeproject
cmake ../ -GXcode
cd ..


