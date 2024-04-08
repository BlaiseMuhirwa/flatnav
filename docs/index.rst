.. FlatNav documentation master file, created by
   sphinx-quickstart on Mon Apr  8 22:05:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FlatNav's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



FlatNav Module
==============

.. automodule:: flatnav.index
    :members:
    :undoc-members:
    :show-inheritance:

Index Factory
-------------

The `index` submodule provides factory functions to create index objects with specific configurations.

.. autofunction:: flatnav.index.index_factory

L2Index Class
-------------

The `L2Index` class represents an index with L2 distance metric.

.. autoclass:: flatnav.index.L2Index
    :members:

IPIndex Class
-------------

The `IPIndex` class represents an index with inner product metric.

.. autoclass:: flatnav.index.IPIndex
    :members:

