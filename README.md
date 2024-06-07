# Linear Algebra

The purpose of this repository is to compare the computational times of various functions in Numpy and Scipy. 

The generated data was created following an exponential distribution with an arbitrary beta vector and Gaussian noise when applicable. All data are normalized.

The results show that for matrices of sizes 1000 by 1000, 1000 by 500, and 500 by 1000, Numpy is more efficient for all the functions studied. Numpy not only shows better computational times but also greater precision, as the time taken for each simulation is more consistent. In contrast, Scipy exhibits higher variance with occasional spikes in computation time

## Least squares

|Matrix Size|Numpy time (s)|Scipy time (s)|
|-|-|-|
|1000 x 1000|0.9|1-3|
|1000 x 500|0.3|0.4-2|
|500 x 1000|0.3|0.5-2|

## Solver

|Matrix Size|Numpy time (s)|Scipy time (s)|
|-|-|-|
|1000 x 1000|0.1|0.4|

## SVD

|Matrix Size|Numpy time (s)|Scipy time (s)|
|-|-|-|
|1000 x 1000|1.1|1.5|
|1000 x 500|0.4|0.5-3|
|500 x 1000|0.5|0.6-2|

## QR 

|Matrix Size|Numpy time (s)|Scipy time (s)|
|-|-|-|
|1000 x 1000|0.3|0.4|
|1000 x 500|0.15|0.3|
|500 x 1000|0.11|0.25|

## Pinv

|Matrix Size|Numpy time (s)|Scipy time (s)|
|-|-|-|
|1000 x 1000|1.3|1.8|
|1000 x 500|0.4|0.7|
|500 x 1000|0.5|0.6|

## Inversion methods

In this section, four different techniques for inverting a square matrix (1000 x1000) will be demonstrated : 
- Invert function
- Moore Penrose pseudo-invert (SVD)
- QR Decomposition
- LU Decomposition

Time computation
|Condition Number|inv|pinv|qr|lu|
|-|-|-|-|-|
e3|0.26|1.25|0.66|0.07|
e10|0.27|1.2|0.69|0.44|
e13|0.19|1.3|0.7|0.1|
e16|0.13|1.2|0.62|0.11|


L2 norm between Identity Maxtrix and the matrix product of X and the calculated invert.
|Condition Number|inv|pinv|qr|lu|
|-|-|-|-|-|
e3|e-22|e-24|e-25|e-22|
e10|e-7|e-11|e-10|e-7|
e13|e-1|e-3|e-4|e-1|
e16|e4|1|e3|e4|