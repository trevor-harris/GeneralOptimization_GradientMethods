# Steepest descent with the fixed number of iterations, function argument is a scalar
####################################################################################

# f - function that calculates and returns the value of f at any given point x
# fprime - function that calculates and returns the derivative of f at any given point x
# x0 - initial starting point
# alpha - positive number, step size
# nIter - positive integer, number of iterations
SteepestDescent <- function(f, fprime, x0, alpha, nIter){
  
  # Initialize storage for iterations and function values
  fvec = rep(f(x0), nIter + 1) # nIter + 1 so that the starting point is saved in addition
  xvec = rep(x0, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  
  # Perform steepest descent update for nIter iterations
  for (i in 1:nIter){
    # At each iteration, update current value of x
    # and save the new function value
    # [ToDo] fill in
    # Steepest descent update
    xvec[i + 1] = xvec[i] - alpha * fprime(xvec[i])
    # Function value
    fvec[i + 1] = f(xvec[i + 1])
  }
  
  # Return the vector of x values, as well as the vector of function values across iterations, including the starting point (both length nIter + 1)
  return(list(xvec = xvec, fvec = fvec))
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
  
  # Initialize storage for iterations and function values
  p = length(x0)
  fvec = rep(f(x0, ...), nIter + 1) # nIter + 1 so that the starting point is saved in addition
  xmat = matrix(x0, p, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  
  # Perform steepest descent update for nIter iterations
  for (i in 1:nIter){
    # At each iteration, update current value of x and save the new function value
    # [ToDo] fill in
    # Steepest descent update
    xmat[ , i + 1] = xmat[ , i] - alpha * fgradient(xmat[ , i], ...)
    # Function value
    fvec[i + 1] = f(xmat[ , i + 1], ...)

  }
  
  # Return the matrix of x values, as well as the vector of function values across iterations, including the starting point (both have nIter + 1 elements, for x put them in columns)
  return(list(xmat = xmat, fvec = fvec))
}
