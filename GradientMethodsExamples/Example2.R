# Binary logistic regression example
###########################################################

# Source out steepest descent functions
source("SteepestDescent.R")

# Source our logistic functions
source("FunctionsBinaryLogistic.R")

# Initialize response and covariates
########################################################################
y = c(1, 1, 0, 0, 1, 0, 1, 0, 0, 0) # response
n = length(y) # number of samples
x1 = c(8, 14, -7, 6, 5, 6, -5, 1, 0, -17) # covariates for 1st variable
X = cbind(rep(1, n), x1) # add a column of 1s for the intercept

# Apply vector form of steepest descent
#######################################################################

beta_init = c(0, 0.2) # initial starting value
alpha = 0.015 # step size
nIter = 15 # number of iterations

out_small <- SteepestDescentVec(f = logistic_objective,
                                fgradient = logistic_gradient,
                                x0 = beta_init, alpha = alpha,
                                nIter = nIter, X = X, y = y)

plot(0:nIter, out_small$fvec, type = 'o', xlab = "Iteration", ylab = "f(beta)")


# Apply customized steepest descent solver
#######################################################################

beta_init = c(0, 0.2) # initial starting value
alpha = 0.015 # step size
nIter = 15 # number of iterations

out_small2 <- SteepestDescentBinLogistic(X = X, y = y, beta_init = beta_init, alpha = alpha, nIter = nIter)

plot(0:nIter, out_small2$fvec, type = 'o', xlab = "Iteration", ylab = "f(beta)")

# Compare the two solvers
#######################################################################
# Return the same answer
sum(abs(out_small2$fvec - out_small$fvec))

# In terms of speed
library(microbenchmark)
microbenchmark(
  SteepestDescentVec(f = logistic_objective, fgradient = logistic_gradient, x0 = beta_init, alpha = alpha, nIter = nIter, X = X, y = y),
  SteepestDescentBinLogistic(X = X, y = y, beta_init = beta_init, alpha = alpha, nIter = nIter),
  times = 10
)

# Different starting point and step sizes
###########################################
beta_init = c(0, 0.32) # initial starting value
alpha = 0.32 # step size
nIter = 15 # number of iterations

out_medium <- SteepestDescentBinLogistic(X = X, y = y, beta_init = beta_init, alpha = alpha, nIter = nIter)

plot(0:nIter, out_medium$fvec, type = 'o', xlab = "Iteration", ylab = "f(beta)")

#######################################################################
# Apply Newton's method
#######################################################################

beta_init = c(0, 0.2) # initial starting value
nIter = 15 # number of iterations

out_Newton1 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init, nIter = nIter)

plot(0:nIter, out_Newton1$fvec, type = 'o', xlab = "Iteration", ylab = "f(beta)")

# Different starting point
#######################################################################
beta_init = c(0.33, 0.33) # initial starting value
nIter = 15 # number of iterations

out_Newton2 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init,  nIter = nIter)

plot(0:nIter, out_Newton2$fvec, type = 'o', xlab = "Iteration", ylab = "f(beta)")


# Same starting point as before, but use damping with eta
#######################################################################
beta_init = c(0.33, 0.33) # initial starting value
nIter = 15 # number of iterations

out_Newton3 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init, nIter = nIter, eta = 0.1)
plot(0:nIter, out_Newton2$fvec, type = 'o', xlab = "Iteration", ylab = "f(beta)", ylim = c(0, 25))
lines(0:nIter, out_Newton3$fvec, type = 'o', col = "red")

plot(0:nIter, out_Newton2$beta_mat[1, ], type = 'o', xlab = "Iteration", ylab = "beta_1", ylim = c(-25, 25))
lines(0:nIter, out_Newton3$beta_mat[1, ], type = 'o', col = "red")


# New starting points that are further away from the truth, but use damping with eta
#######################################################################
beta_init1 = c(0, 1) 
beta_init2 = c(0, 5)
beta_init3 = c(0, 10)
nIter = 30
out1 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init1, nIter = 30, eta = 0.1)
out2 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init2, nIter = 30, eta = 0.1)
out3 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init3, nIter = 30, eta = 0.1)

plot(0:nIter, out1$beta_mat[2, ], type = 'o', xlab = "Iteration", ylab = expression(beta_2), ylim = c(-10, 10))
lines(0:nIter, out2$beta_mat[2, ], type = 'o', col = "red")
lines(0:nIter, out3$beta_mat[2, ], type = 'o', col = "blue")


# Same starting point, use ridge regularization
#######################################################################
out1 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init1, nIter = 30, eta = 0.1, lambda = 1)
out2 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init2, nIter = 30, eta = 0.1, lambda = 1)
out3 <- NewtonBinLogistic(X = X, y = y, beta_init = beta_init3, nIter = 30, eta = 0.1, lambda = 1)

plot(0:nIter, out1$beta_mat[2, ], type = 'o', xlab = "Iteration", ylab = expression(beta_2), ylim = c(-10, 10))
lines(0:nIter, out2$beta_mat[2, ], type = 'o', col = "red")
lines(0:nIter, out3$beta_mat[2, ], type = 'o', col = "blue")