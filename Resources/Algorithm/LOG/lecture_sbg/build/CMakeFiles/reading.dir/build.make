# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/build

# Include any dependencies generated for this target.
include CMakeFiles/reading.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/reading.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/reading.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reading.dir/flags.make

CMakeFiles/reading.dir/src/read_dvl.cpp.o: CMakeFiles/reading.dir/flags.make
CMakeFiles/reading.dir/src/read_dvl.cpp.o: ../src/read_dvl.cpp
CMakeFiles/reading.dir/src/read_dvl.cpp.o: CMakeFiles/reading.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/reading.dir/src/read_dvl.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reading.dir/src/read_dvl.cpp.o -MF CMakeFiles/reading.dir/src/read_dvl.cpp.o.d -o CMakeFiles/reading.dir/src/read_dvl.cpp.o -c /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/src/read_dvl.cpp

CMakeFiles/reading.dir/src/read_dvl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reading.dir/src/read_dvl.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/src/read_dvl.cpp > CMakeFiles/reading.dir/src/read_dvl.cpp.i

CMakeFiles/reading.dir/src/read_dvl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reading.dir/src/read_dvl.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/src/read_dvl.cpp -o CMakeFiles/reading.dir/src/read_dvl.cpp.s

# Object files for target reading
reading_OBJECTS = \
"CMakeFiles/reading.dir/src/read_dvl.cpp.o"

# External object files for target reading
reading_EXTERNAL_OBJECTS =

reading: CMakeFiles/reading.dir/src/read_dvl.cpp.o
reading: CMakeFiles/reading.dir/build.make
reading: CMakeFiles/reading.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reading"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reading.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reading.dir/build: reading
.PHONY : CMakeFiles/reading.dir/build

CMakeFiles/reading.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reading.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reading.dir/clean

CMakeFiles/reading.dir/depend:
	cd /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/build /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/build /home/am/Documents/Cours/Semestre_5/Guerledan_recalage/Resources/Algorithm/LOG/lecture_sbg/build/CMakeFiles/reading.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reading.dir/depend

