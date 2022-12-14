# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_SOURCE_DIR = "/home/dsp520/Desktop/TM test"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/dsp520/Desktop/TM test"

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components

.PHONY : list_install_components/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: install/strip

.PHONY : install/strip/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: install/local

.PHONY : install/local/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start "/home/dsp520/Desktop/TM test/CMakeFiles" "/home/dsp520/Desktop/TM test/CMakeFiles/progress.marks"
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start "/home/dsp520/Desktop/TM test/CMakeFiles" 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named test_tm_driver

# Build rule for target.
test_tm_driver: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test_tm_driver
.PHONY : test_tm_driver

# fast build rule for target.
test_tm_driver/fast:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/build
.PHONY : test_tm_driver/fast

src/tm_communication.o: src/tm_communication.cpp.o

.PHONY : src/tm_communication.o

# target to build an object file
src/tm_communication.cpp.o:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_communication.cpp.o
.PHONY : src/tm_communication.cpp.o

src/tm_communication.i: src/tm_communication.cpp.i

.PHONY : src/tm_communication.i

# target to preprocess a source file
src/tm_communication.cpp.i:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_communication.cpp.i
.PHONY : src/tm_communication.cpp.i

src/tm_communication.s: src/tm_communication.cpp.s

.PHONY : src/tm_communication.s

# target to generate assembly for a file
src/tm_communication.cpp.s:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_communication.cpp.s
.PHONY : src/tm_communication.cpp.s

src/tm_driver.o: src/tm_driver.cpp.o

.PHONY : src/tm_driver.o

# target to build an object file
src/tm_driver.cpp.o:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_driver.cpp.o
.PHONY : src/tm_driver.cpp.o

src/tm_driver.i: src/tm_driver.cpp.i

.PHONY : src/tm_driver.i

# target to preprocess a source file
src/tm_driver.cpp.i:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_driver.cpp.i
.PHONY : src/tm_driver.cpp.i

src/tm_driver.s: src/tm_driver.cpp.s

.PHONY : src/tm_driver.s

# target to generate assembly for a file
src/tm_driver.cpp.s:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_driver.cpp.s
.PHONY : src/tm_driver.cpp.s

src/tm_print.o: src/tm_print.cpp.o

.PHONY : src/tm_print.o

# target to build an object file
src/tm_print.cpp.o:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_print.cpp.o
.PHONY : src/tm_print.cpp.o

src/tm_print.i: src/tm_print.cpp.i

.PHONY : src/tm_print.i

# target to preprocess a source file
src/tm_print.cpp.i:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_print.cpp.i
.PHONY : src/tm_print.cpp.i

src/tm_print.s: src/tm_print.cpp.s

.PHONY : src/tm_print.s

# target to generate assembly for a file
src/tm_print.cpp.s:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_print.cpp.s
.PHONY : src/tm_print.cpp.s

src/tm_robot_state_rt.o: src/tm_robot_state_rt.cpp.o

.PHONY : src/tm_robot_state_rt.o

# target to build an object file
src/tm_robot_state_rt.cpp.o:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_robot_state_rt.cpp.o
.PHONY : src/tm_robot_state_rt.cpp.o

src/tm_robot_state_rt.i: src/tm_robot_state_rt.cpp.i

.PHONY : src/tm_robot_state_rt.i

# target to preprocess a source file
src/tm_robot_state_rt.cpp.i:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_robot_state_rt.cpp.i
.PHONY : src/tm_robot_state_rt.cpp.i

src/tm_robot_state_rt.s: src/tm_robot_state_rt.cpp.s

.PHONY : src/tm_robot_state_rt.s

# target to generate assembly for a file
src/tm_robot_state_rt.cpp.s:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/src/tm_robot_state_rt.cpp.s
.PHONY : src/tm_robot_state_rt.cpp.s

test_tm_driver.o: test_tm_driver.cpp.o

.PHONY : test_tm_driver.o

# target to build an object file
test_tm_driver.cpp.o:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/test_tm_driver.cpp.o
.PHONY : test_tm_driver.cpp.o

test_tm_driver.i: test_tm_driver.cpp.i

.PHONY : test_tm_driver.i

# target to preprocess a source file
test_tm_driver.cpp.i:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/test_tm_driver.cpp.i
.PHONY : test_tm_driver.cpp.i

test_tm_driver.s: test_tm_driver.cpp.s

.PHONY : test_tm_driver.s

# target to generate assembly for a file
test_tm_driver.cpp.s:
	$(MAKE) -f CMakeFiles/test_tm_driver.dir/build.make CMakeFiles/test_tm_driver.dir/test_tm_driver.cpp.s
.PHONY : test_tm_driver.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... install"
	@echo "... list_install_components"
	@echo "... rebuild_cache"
	@echo "... test_tm_driver"
	@echo "... install/strip"
	@echo "... install/local"
	@echo "... edit_cache"
	@echo "... src/tm_communication.o"
	@echo "... src/tm_communication.i"
	@echo "... src/tm_communication.s"
	@echo "... src/tm_driver.o"
	@echo "... src/tm_driver.i"
	@echo "... src/tm_driver.s"
	@echo "... src/tm_print.o"
	@echo "... src/tm_print.i"
	@echo "... src/tm_print.s"
	@echo "... src/tm_robot_state_rt.o"
	@echo "... src/tm_robot_state_rt.i"
	@echo "... src/tm_robot_state_rt.s"
	@echo "... test_tm_driver.o"
	@echo "... test_tm_driver.i"
	@echo "... test_tm_driver.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

