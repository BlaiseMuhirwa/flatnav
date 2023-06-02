
# Format all header files with clang-format 
find . -iname *.h -o -iname *.cpp | xargs clang-format -i 

# Format CMakeLists.txt file 
cmake-format -i CMakeLists.txt 