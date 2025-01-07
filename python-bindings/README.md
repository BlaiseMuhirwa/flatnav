

## Building the FlatNav Python Library 


First, if you are on a Linux machine (e.g. Ubuntu), ensure that you have the header files and static libraries
for python dev. In addition, if you do not have the C++ compiler, you will need to install that first. 

```shell
> sudo apt-get install python3-dev build-essential
```

To build the wheel file and pip-install it, run

```shell
> cd flatnav_python
> poetry install --no-dev
> ./install_flatnav.sh 
```