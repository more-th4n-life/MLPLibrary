Setup & Installation
==========================

This section will give instructions to install and start using MLPLibrary.

Getting Started
----------------------------

First, visit the code repository at ``https://github.com/more-th4n-life/MLPLibrary``
and clone it.

You may need to install some dependencies prior to using MLPLibrary. Some may already be 
installed in your dev environment. It is assumed that you have Numpy and Python already
installed. Upon installation of MLPLibrary, these dependencies should automatically be 
installed as they are stated within the ``setup.py`` file within the src directory. 


Installing the MLPLibrary
-----------------------------------------

- Go `to the repository`_ and clone or download the code.
- Navigate to ``MLPLibrary`` root folder and install via ``pip install .``. This 
  installs the backend library as a python module on your system to use locally. 

Optional: Documentation Generation
---------------------------------------

To build the latest documentation with Sphinx - you need to install the Sphinx
documentation tool. 

- First, install `MLPLibrary` as per above instructions.

- Follow instructions at `Sphinx official install instructions`_ for your OS.

- Install the documentation theme in your virtual environment with
  ``pip install sphinx-rtd-theme``

- Navigate to ``MLPLibrary/docs`` to generate the latest API documentation
  modules with command ``sphinx-apidoc -fo source/ ../network``
      
- Now build the html docs with ``make html``
  
- The resultant docs html homepage will be found at ``MLPlibrary/docs/build/html/index.html``

.. _to the repository: https://github.com/more-th4n-life/MLPLibrary
.. _Sphinx official install instructions: https://www.sphinx-doc.org/en/master/usage/installation.html