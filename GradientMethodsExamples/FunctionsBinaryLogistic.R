# Objective calculation for binary logistic regression
########################################################################
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_objective <- function(beta, X, y){
  # [ToDo] Compute value of objective function for binary logistic regression using current value of X, y and beta
  Xb = X %*% beta # n by 1
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
  # Probabilities for given b
  prob = exp(X %*% beta) 
  prob = prob / (1 + prob)
  
  # Gradient given probabilities
  gradient = crossprod(X, prob - y)
  
  return(gradient)
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
  
  # [ToDo]Calculate current objective value
  Xb = X %*% beta_init # n by 1
  fvec[1] = sum(- y * Xb + log(1 + exp(Xb)))

  
  # Perform steepest descent update for nIter iterations
  for (i in 1:nIter){
    # [ToDo]At each iteration, 
    # calculate gradient value
    # Probabilities for given b
    prob = exp(Xb) 
    prob = prob / (1 + prob)
    
    # Gradient given probabilities
    gradient = crossprod(X, prob - y)
    
    # update x
    beta_mat[ , i + 1] = beta_mat[ , i] - alpha * gradient
    
    # calculate current function value
    Xb = X %*% beta_mat[, i + 1] # n by 1
    fvec[i + 1] = sum(- y * Xb + log(1 + exp(Xb)))
  }
  
  # Return the matrix of x values, as well as the vector of function values across iterations, including the starting point (both have nIter + 1 elements, for beta_mat put them in columns)
  return(list(beta_mat = beta_mat, fvec = fvec))
}


# Write down customized solver of Newton's method on binary logistic to avoid recalculating extra things
########################################################################
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
# beta_init - initial starting vector (dimension p)
# nIter - positive integer, number of iterations
# eta - positive scalar, learning rate, 1 by default (will use later for damping)
# lambda - positive scalar, ridge penalty parameter, 0 by default (will use later for ridge)
NewtonBinLogistic <- function(X, y, beta_init, nIter, eta = 1, lambda = 0){
  
  # [ToDo] Initialize storage for iterations and function values
  p = length(beta_init)
  n = nrow(X)
  fvec = rep(0, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  beta_mat = matrix(beta_init, p, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  
  # [ToDo]Calculate current objective value
  Xb = X %*% beta_init # n by 1
  fvec[1] = sum(- y * Xb + log(1 + exp(Xb))) + lambda * sum(beta_init^2)/2
  
  # Perform Newton update for nIter iterations
  for (i in 1:nIter){
    # [ToDo]At each iteration, calculate gradient value, Hessian, update x, calculate current function value
    # calculate gradient value
    # Probabilities for given b
    prob = exp(Xb) 
    prob = prob / (1 + prob)
    
    # Gradient given probabilities
    gradient = crossprod(X, prob - y) + lambda * beta_mat[ , i]
    
    # update x (need to change for Newton's method)
    # Calculate hessian
    # W = diag(prob * (1 - prob)) # n by n
    w = as.vector(prob * (1 - prob))
    hessian = crossprod(X, X * w) + lambda * diag(p) # p by p
    # Do a step
    beta_mat[ , i + 1] = beta_mat[ , i] - eta * solve(hessian, gradient)
    
    # calculate current function value
    Xb = X %*% beta_mat[, i + 1] # n by 1
    fvec[i + 1] = sum(- y * Xb + log(1 + exp(Xb))) + 
                  lambda * sum(beta_mat[ , i + 1]^2)/2
  }
  
  # Return the matrix of x values, as well as the vector of function values across iterations, including the starting point (both have nIter + 1 elements, for x put them in columns)
  return(list(beta_mat = beta_mat, fvec = fvec))
}