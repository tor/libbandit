#LibBandit

LibBandit is a C++ library designed for efficiently simulating multi-armed bandit algorithms.

Currently the following algorithms are implemented:
* UCB
* Optimally confident UCB
* Almost optimally confident UCB
* Thompson sampling (Gaussian prior)
* MOSS
* Finite-horizon Gittins index (Gaussian/Gaussian model/prior)
* An approximation of the finite-horizon Gittins index
* Bayesian optimal for two arms (Gaussian/Gaussian model/prior)

Defining new noise models is as simple as extending a base class and implementing the reward function.


##Compiling

You will need a C++11 compliant compiler such as g++ 4.8 or clang 5.

LibBandit uses the Scons build system. With this installed you should be able to compile all sources by typing `scons`


##Using the Library

LibBandit is easy to use. See the examples/ folder.


##Gittins Index

The library includes code for efficiently generating Gittins indices for a Gaussian prior and noise model. Included is a precomputed
table of indices for horizons up to 5,000. See the examples/ folder for details on how to use this data. 

To compute the indices yourself use `makegittins build <file> <horizon> <tolerance> <maxthreads>`

The tolerance should be chosen as small as possible. The pre-computed table used tolerance = 0.000005.

You can lookup the Gittins index in a table with `makegittins lookup <file> <horizon> <T>` where <horizon> is the number of rounds
remaining and <T> is the number of samples from that arm.

A larger pre-computed table for horizon 10,000 and tolerance 0.000005 is available for download from http://downloads.tor-lattimore.com/gittins/10000.zip.



##Contributing

If you implement a new algorithm please (a) test it against existing algorithms and (b) contact me to become a 
contributor so others can easily test against your algorithm.



