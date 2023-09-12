# One-dimensional optimization of a convex function

# Function definition 
f <- function(x){
  (x - 50)^2 + exp(x)/50
}

# Function first derivative
fprime <- function(x){
  2 * x - 100 + exp(x)/50
}

# Function second derivative
fdoubleprime <- function(x){
  2 + exp(x)/50
}

# Basic function visualization
plot(seq(-10, 12, length = 100), f(seq(-10, 12, length = 100)), type = "l", xlab = "x", ylab = "f(x)") 

# Use built in solvers to minimize f(x)
########################################################
# Using optimize to minimize f(x)
optimize(f, interval = c(-100, 100))

# Using uniroot to find roots to gradient equation
uniroot(fprime, interval = c(-100, 100))

# Compare the two in terms of speed - optimize is faster here
library(microbenchmark)
microbenchmark(
  optimize(f, interval = c(-100, 100)),
  uniroot(fprime, interval = c(-100, 100))
)

# Use steepest descent with different step sizes
########################################################

# Source our steepest descent functions
source("SteepestDescent.R")

# Number of iterations
nIter = 30
# Small step size
alpha = 0.001

out_small <- SteepestDescent(f, fprime, x0 = 0, alpha = alpha, nIter = nIter)

plot(0:nIter, out_small$xvec, type = 'o', xlab = "Iteration t", ylab = "Value of xt")

plot(0:nIter, out_small$fvec, type = 'o', xlab = "Iteration t", ylab = "Value of f(xt)")

# Medium step size
alpha = 0.01

out_medium <- SteepestDescent(f, fprime, x0 = 0, alpha = alpha, nIter = nIter)

plot(0:nIter, out_medium$xvec, type = 'o', xlab = "Iteration t", ylab = "Value of xt")

plot(0:nIter, out_medium$fvec, type = 'o', xlab = "Iteration t", ylab = "Value of f(xt)")

# Large step size
alpha = 0.03

out_large <- SteepestDescent(f, fprime, x0 = 0, alpha = alpha, nIter = nIter)

plot(0:nIter, out_large$xvec, type = 'o', xlab = "Iteration t", ylab = "Value of xt")

plot(0:nIter, out_large$fvec, type = 'o', xlab = "Iteration t", ylab = "Value of f(xt)")


# Use Newton method
########################################################

# Source Newton's method functions
source("NewtonsMethod.R")

nIter = 30

# Starting point x0 = 5
out_Newton1 <- NewtonsMethod(f, fprime, fdoubleprime, x0 = 5, nIter = nIter)
plot(0:nIter, out_Newton1$xvec, type = 'o', xlab = "Iteration t", ylab = "Value of xt")
plot(0:nIter, out_Newton1$fvec, type = 'o', xlab = "Iteration t", ylab = "Value of f(xt)")

# Starting point x0 = 8
out_Newton2 <- NewtonsMethod(f, fprime, fdoubleprime, x0 = 8, nIter = nIter)
plot(0:nIter, out_Newton2$xvec, type = 'o', xlab = "Iteration t", ylab = "Value of xt")
plot(0:nIter, out_Newton2$fvec, type = 'o', xlab = "Iteration t", ylab = "Value of f(xt)")

# Starting point x0 = 10
out_Newton3 <- NewtonsMethod(f, fprime, fdoubleprime, x0 = 10, nIter = nIter)
plot(0:nIter, out_Newton3$xvec, type = 'o', xlab = "Iteration t", ylab = "Value of xt")
plot(0:nIter, out_Newton3$fvec, type = 'o', xlab = "Iteration t", ylab = "Value of f(xt)")
