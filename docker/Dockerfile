
FROM quay.io/pypa/manylinux2014_x86_64:latest

# Install gcc
RUN yum install -y gcc \
    devtoolset-11-gcc \
    devtoolset-11-gcc-c++

COPY ./entrypoint.sh . 
ENTRYPOINT ["/entrypoint.sh"]