# Code (Don't commit yet)

SteepestDescent <- function(f, fprime, x0, alpha, nIter){
  # Perform steepest descent update for nIter iterations
  fvec = rep(f(x0), nIter + 1)
  xvec = rep(x0, nIter + 1)
  
  # At each iteration, save the function value f(x_t)
  for (i in 1:nIter){
    xt = x0 - alpha * fprime(x0) # steepest descent update
    fvec[i + 1] = f(xt) # evaluate value of f at point xt
    xvec[i + 1] = xt # save xt
    x0 = xt # make it a new starting point
  }
  # Return the vector of x values, as well as the vector of function values across iterations
  return(list(xvec = xvec, fvec  = fvec))
}

SteepestDescentVec <- function(f, fgradient, x0, alpha, nIter, ...){
  fvec = rep(f(x0, ...), nIter + 1)
  p = length(x0)
  xmat = matrix(x0, p, nIter + 1)
  for (i in 1:nIter){
    xt = x0 - alpha * fgradient(x0, ...)
    fvec[i + 1] = f(xt, ...)
    xmat[ , i + 1] = xt
    x0 = xt
  }
  return(list(xmat = xmat, fvec  = fvec))
}

NewtonsMethod <- function(f, fprime, fdoubleprime, x0, nIter){
  fvec = rep(f(x0), nIter + 1)
  xvec = rep(x0, nIter + 1)
  for (i in 1:nIter){
    xt = x0 - fprime(x0) / fdoubleprime(x0)
    fvec[i + 1] = f(xt)
    xvec[i + 1] = xt
    x0 = xt
  }
  return(list(xvec = xvec, fvec  = fvec))
}


# Objective calculation for binary logistic regression
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_objective <- function(beta, X, y){
  # [ToDo] Compute value of objective function for binary logistic regression using current value of X, y and beta
  Xb = X %*% beta
  obj = sum(-y * Xb + log(1 + exp(Xb)))
  return(obj)
}

# Gradient calculation for binary logistic regression
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_gradient <- function(beta, X, y){
  # [ToDo] Compute gradient for binary logistic regression problem using current values of X, y and beta
  pbeta = exp(X %*% beta)
  pbeta = pbeta/(1 + pbeta)
  gradient = crossprod(X, pbeta - y)
  return(gradient)
}


