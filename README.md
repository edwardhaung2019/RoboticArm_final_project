#test_tm_driver
TM test without using ROS

## Installation
1. First, modify the file path, include 'CMakeCache.txt', 'cmake_install.cmake'.
2. Open the terminal, and go in to the path of this folder.
3. ```$ cmake .``` and ```$ make``` to build this test code

## Usage
To start this test program:
```
$ ./test_tm_driver 192.168.0.10
```
during the execution
* input ```start``` to connect to techman robot
* input ```halt``` to disconnect to techman robot
* input ```quit``` to exit the program

if the connection is success
* input ```datart``` to print all robot_state_rt once
* input ```show``` to print robot_state cyclic, type ```1, 2, ..., 9, 0``` to change data type, type ```q``` to leave the loop
* there are other commands: ```clear```, ```movjabs```, etc

TODO: the usage of all robot commands...
