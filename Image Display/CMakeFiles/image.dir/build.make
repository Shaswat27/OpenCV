# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shaswat/OpenCV

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shaswat/OpenCV

# Include any dependencies generated for this target.
include CMakeFiles/image.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/image.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/image.dir/flags.make

CMakeFiles/image.dir/image.cpp.o: CMakeFiles/image.dir/flags.make
CMakeFiles/image.dir/image.cpp.o: image.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/shaswat/OpenCV/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/image.dir/image.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/image.dir/image.cpp.o -c /home/shaswat/OpenCV/image.cpp

CMakeFiles/image.dir/image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/image.dir/image.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/shaswat/OpenCV/image.cpp > CMakeFiles/image.dir/image.cpp.i

CMakeFiles/image.dir/image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/image.dir/image.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/shaswat/OpenCV/image.cpp -o CMakeFiles/image.dir/image.cpp.s

CMakeFiles/image.dir/image.cpp.o.requires:
.PHONY : CMakeFiles/image.dir/image.cpp.o.requires

CMakeFiles/image.dir/image.cpp.o.provides: CMakeFiles/image.dir/image.cpp.o.requires
	$(MAKE) -f CMakeFiles/image.dir/build.make CMakeFiles/image.dir/image.cpp.o.provides.build
.PHONY : CMakeFiles/image.dir/image.cpp.o.provides

CMakeFiles/image.dir/image.cpp.o.provides.build: CMakeFiles/image.dir/image.cpp.o

# Object files for target image
image_OBJECTS = \
"CMakeFiles/image.dir/image.cpp.o"

# External object files for target image
image_EXTERNAL_OBJECTS =

image: CMakeFiles/image.dir/image.cpp.o
image: CMakeFiles/image.dir/build.make
image: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
image: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
image: CMakeFiles/image.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable image"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/image.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/image.dir/build: image
.PHONY : CMakeFiles/image.dir/build

CMakeFiles/image.dir/requires: CMakeFiles/image.dir/image.cpp.o.requires
.PHONY : CMakeFiles/image.dir/requires

CMakeFiles/image.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/image.dir/cmake_clean.cmake
.PHONY : CMakeFiles/image.dir/clean

CMakeFiles/image.dir/depend:
	cd /home/shaswat/OpenCV && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shaswat/OpenCV /home/shaswat/OpenCV /home/shaswat/OpenCV /home/shaswat/OpenCV /home/shaswat/OpenCV/CMakeFiles/image.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/image.dir/depend

