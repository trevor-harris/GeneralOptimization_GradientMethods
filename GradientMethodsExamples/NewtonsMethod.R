# Newton's method with the fixed number of iterations, function argument is a scalar
#####################################################################################

# f - function that calculates and returns the value of f at any given point x
# fprime - function that calculates and returns the derivative of f at any given point x
# fdoubleprime - function that calculates and returns the second derivative of f at any given point x
# x0 - initial starting point
# nIter - positive integer, number of iterations

NewtonsMethod <- function(f, fprime, fdoubleprime, x0, nIter){
  
  # Initialize storage for iterations and function values
  fvec = rep(f(x0), nIter + 1)
  xvec = rep(x0, nIter + 1)
  
  # Perform Newton's update for nIter iterations
  for (i in 1:nIter){
    # At each iteration, update current value of x and save the new function value
    # [ToDo] fill in
    # Newton's method update
    xvec[i + 1] = xvec[i] - fprime(xvec[i]) / fdoubleprime(xvec[i])
    # Function value
    fvec[i + 1] = f(xvec[i + 1])
  }
  
  # Return the vector of x values, as well as the vector of function values across iterations, including the starting point (both length nIter + 1)
  return(list(xvec = xvec, fvec  = fvec))
}