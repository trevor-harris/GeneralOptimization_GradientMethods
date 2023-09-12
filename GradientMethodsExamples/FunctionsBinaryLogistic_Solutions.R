# Objective calculation for binary logistic regression
########################################################################
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_objective <- function(beta, X, y){
  # [ToDo] Compute value of objective function for binary logistic regression using current value of X, y and beta
  Xb = X %*% beta
  obj = sum(-y * Xb + log(1 + exp(Xb)))/length(y)
  return(obj)
}

# Gradient calculation for binary logistic regression
########################################################################
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_gradient <- function(beta, X, y){
  # [ToDo] Compute gradient for binary logistic regression problem using current values of X, y and beta
  pbeta = exp(X %*% beta)
  pbeta = pbeta/(1 + pbeta)
  gradient = crossprod(X, pbeta - y)/length(y)
  return(gradient)
}

# Calculation for gradient and objective at once
########################################################################
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_both <- function(beta, X, y){
  # [ToDo] Compute both objective and gradient for binary logistic regression problem using current values of X, y and beta
  Xb = X %*% beta
  pbeta = exp(Xb) # just numerator
  obj = sum(-y * Xb + log(1 + pbeta)) # use numerator for objective
  pbeta = pbeta/(1 + pbeta) # adjust so that actual probability
  gradient = crossprod(X, pbeta - y) # calculate gradient
  # Return the objective value and the gradient value
  return(list(obj = obj, gradient = gradient))
}

# Write down customized solver of steepest descent on binary logistic to avoid recalculating extra things
########################################################################
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
# beta_init - initial starting vector (dimension p)
# alpha - positive scalar, learning rate
# nIter - positive integer, number of iterations
SteepestDescentBinLogistic <- function(X, y, beta_init, alpha, nIter){
  # [ToDo] Initialize storage for iterations and function values
  p = length(beta_init)
  fvec = rep(0, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  beta_mat = matrix(beta_init, p, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  
  # Calculate current objective value
  Xb = X %*% beta_init
  pbeta = exp(Xb) # just numerator
  fvec[1] = sum(-y * Xb + log(1 + pbeta)) # use numerator for objective
  pbeta = pbeta/(1 + pbeta) # adjust so that actual probability
  
  # Perform steepest descent update for nIter iterations
  for (i in 1:nIter){
    # At each iteration
    # Calculate gradient value and update x
    beta = beta_init - alpha * crossprod(X, pbeta - y)
    
    # Update the objective
    Xb = X %*% beta
    pbeta = exp(Xb) # just numerator
    fvec[i + 1] = sum(-y * Xb + log(1 + pbeta)) # use numerator for objective
    
    # Update pbeta for next round
    pbeta = pbeta/(1 + pbeta) # adjust so that actual probability
    beta_init = beta
    beta_mat[, i + 1] = beta
  }
  
  # Return the matrix of x values, as well as the vector of function values across iterations, including the starting point (both have nIter + 1 elements, for x put them in columns)
  return(list(beta_mat = beta_mat, fvec = fvec))
}

# Write down customized solver of Newton's method on binary logistic to avoid recalculating extra things
########################################################################
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
# beta_init - initial starting vector (dimension p)
# nIter - positive integer, number of iterations
# eta - positive scalar, learning rate, 1 by default (will use later for damping)
# lambda - positive scalar, ridge penalty, 0 by default (will use later for ridge)
NewtonBinLogistic <- function(X, y, beta_init, nIter, eta = 1, lambda = 0){

  # [ToDo] Initialize storage for iterations and function values
  p = length(beta_init)
  fvec = rep(0, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  beta_mat = matrix(beta_init, p, nIter + 1) # nIter + 1 so that the starting point is saved in addition

  # Calculate current objective value
  Xb = X %*% beta_init
  pbeta = exp(Xb) # just numerator
  fvec[1] = sum(-y * Xb + log(1 + pbeta)) # use numerator for objective
  pbeta = pbeta/(1 + pbeta) # adjust so that actual probability
  
  
  # Perform steepest descent update for nIter iterations
  for (i in 1:nIter){
    # At each iteration, calculate gradient value, Hessian, update x, calculate current function value
    
    # Calculate Hessian value and update beta_mat
    H = crossprod(X,  X * as.vector(pbeta * (1 - pbeta))) + lambda * diag(p)
    beta = beta_init - eta * solve(H, crossprod(X, pbeta - y) + lambda * beta_init)
    
    # Update the objective
    Xb = X %*% beta
    pbeta = exp(Xb) # just numerator
    fvec[i + 1] = sum(-y * Xb + log(1 + pbeta)) # use numerator for objective
    
    # Update pbeta for next round
    pbeta = pbeta/(1 + pbeta) # adjust so that actual probability
    beta_init = beta
    beta_mat[, i + 1] = beta
  }
  
  # Return the matrix of x values, as well as the vector of function values across iterations, including the starting point (both have nIter + 1 elements, for x put them in columns)
  return(list(beta_mat = beta_mat, fvec = fvec))
}


