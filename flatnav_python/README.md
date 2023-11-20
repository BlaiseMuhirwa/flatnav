

## Building the Python Wheel file 


First, if you are on a Linux machine (e.g. Ubuntu), ensure that you have the header files and static libraries
for python dev. To install them on Ubuntu, run 

```shell
> sudo apt-get install python3-dev
```

To build the wheel file, run 

```shell
> poetry run python setup.py bdist_wheel 
```