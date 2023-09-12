# Objective calculation for binary logistic regression
########################################################################
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_objective <- function(beta, X, y){
  # [ToDo] Compute value of objective function for binary logistic regression using current value of X, y and beta
  Xb = X %*% beta # each row x_i'beta, n by 1
  #pbeta = exp(Xb) 
  #pbeta = pbeta / (1 + pbeta)
  #obj = sum((1-y) * log(1 - pbeta) + y * log(pbeta))
  obj = sum(- y * Xb + log(1 + exp(Xb)))
  return(obj)
}

# Gradient calculation for binary logistic regression
########################################################################
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_gradient <- function(beta, X, y){
  # [ToDo] Compute gradient for binary logistic regression problem using current values of X, y and beta
  pbeta = exp(X %*% beta) # numerator
  pbeta = pbeta / (1 + pbeta) #adjust for denominator
  gradient = crossprod(X, pbeta - y) # gradient, p-dim vector
  return(gradient)
}

# Steepest descent with the fixed number of iterations, function argument is a vector
#####################################################################################
# f - function that calculates and returns the value of f at any given vector x
# fgradient - function that calculates and returns the gradient of f at any given vector x
# x0 - initial starting vector
# alpha - positive number, step size
# nIter - positive integer, number of iterations
# ... - other arguments that may be needed for calculation of f and fgradient
SteepestDescentVec <- function(f, fgradient, x0, alpha, nIter, ...){
  
  # [ToDo] Initialize storage for iterations and function values
  p = length(x0)
  fvec = rep(f(x0, ...), nIter + 1) # nIter + 1 so that the starting point is saved in addition
  xmat = matrix(x0, p, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  
  # Perform steepest descent update for nIter iterations
  for (i in 1:nIter){
    # At each iteration, update current value of x and save the new function value
    xnew = x0 - alpha * fgradient(x0, ...) # update step
    fvec[i + 1] = f(xnew, ...) # new objective value
    xmat[ , i + 1] = xnew # current argument
    x0 = xnew # update x0 so the next update is correct
  }
  
  # Return the matrix of x values, as well as the vector of function values across iterations, including the starting point (both have nIter + 1 elements, for x put them in columns)
  return(list(xmat = xmat, fvec = fvec))
}

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

out_small <- SteepestDescentVec(f = logistic_objective, fgradient = logistic_gradient, x0 = beta_init, alpha = alpha, nIter = nIter, X = X, y = y)

plot(0:nIter, out_small$fvec, type = 'o', xlab = "Iteration", ylab = "f(beta)")