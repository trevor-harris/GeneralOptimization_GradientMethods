# Objective calculation for binary logistic regression
########################################################################
# beta - parameter vector of length p
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
logistic_objective <- function(beta, X, y){
  Xb = X %*% beta # each row x_i'beta, n by 1
  obj = sum(- y * Xb + log(1 + exp(Xb)))/length(y)
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
  gradient = crossprod(X, pbeta - y)/length(y) # gradient, p-dim vector
  return(gradient)
}


# Write down customized solver of steepest descent on binary logistic to avoid recalculating extra things
########################################################################
# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
# beta_init - initial starting vector (dimension p)
# alpha - positive scalar, learning rate
# nIter - positive integer, number of iterations
# b - batch size
SteepestDescentBinLogistic <- function(X, y, beta_init, alpha, nIter, b = NULL){
  # [ToDo] Initialize storage for iterations and function values
  p = length(beta_init)
  n = length(y)
  if (is.null(b)){
    b = n # full gradient
  }
  fvec = rep(0, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  beta_mat = matrix(beta_init, p, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  
  # Calculate current objective value
  Xb = X %*% beta_init # each row x_i'beta, n by 1
  pbeta = exp(Xb)
  fvec[1] = sum(- y * Xb + log(1 + pbeta))
  beta = beta_init
  
  # Perform steepest descent update for nIter iterations
  for (i in 1:nIter){
    # At each iteration, calculate gradient value, update x, calculate current function value
    # Calculate gradient value and update x
    if (b == n){
      beta = beta_init - alpha * logistic_gradient(beta, X, y)
    }else{
      Ii = sample(1:n, size = b)
      beta = beta_init - alpha * logistic_gradient(beta, X[Ii, , drop = F], y[Ii, drop = F])
    }
  
    
    # Update the objective
    fvec[i + 1] = logistic_objective(beta, X, y) # use numerator for objective
    
    # Update beta for next round
    beta_init = beta
    beta_mat[, i + 1] = beta
    
    # Decrease step size a bit
    #alpha = alpha * 0.9
  }
  
  # Return the matrix of x values, as well as the vector of function values across iterations, including the starting point (both have nIter + 1 elements, for beta_mat put them in columns)
  return(list(beta_mat = beta_mat, fvec = fvec))
}

# X - n by p matrix of covariates
# y - vector of length n, binary response (0 or 1)
# beta_init - initial starting vector (dimension p)
# alpha - positive scalar, learning rate
# nIter - positive integer, number of iterations
SteepestDescentBinLogisticSAG <- function(X, y, beta_init, alpha, nIter){
  # [ToDo] Initialize storage for iterations and function values
  p = length(beta_init)
  n = length(y)
  fvec = rep(0, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  beta_mat = matrix(beta_init, p, nIter + 1) # nIter + 1 so that the starting point is saved in addition
  
  # Calculate current objective value
  Xb = X %*% beta_init # each row x_i'beta, n by 1
  pbeta = exp(Xb)
  fvec[1] = sum(- y * Xb + log(1 + pbeta))
  beta = beta_init
  
  # Compute current gradient (average of all gradients at the starting point beta_init)
  pbeta = exp(X %*% beta) # numerator (n by 1)
  pbeta = pbeta / (1 + pbeta) #adjust for denominator (n by 1)
  gradient_all = X * matrix(pbeta - y, n, p, byrow = T) # gradient, n by p
  g = colMeans(gradient_all)
  
  # Perform steepest descent update for nIter iterations
  for (i in 1:nIter){
    # At each iteration, calculate gradient value, update x, calculate current function value
    it = sample(1:n, size = 1) #pick one random sample
    # Compute the gradient just at that sample
    pi = exp(X[it, ] * beta)
    pi = pi / (1 + pi)
    gi = X[it, ] * (pi - y[it])
    #
    beta = beta_init - alpha * (gi/n - gradient_all[it, ]/n + g)
    # Update new average
    g = gi/n - gradient_all[it, ]/n + g
    gradient_all[it, ] = gi
  
    # Update the objective
    fvec[i + 1] = logistic_objective(beta, X, y) # use numerator for objective
    
    # Update beta for next round
    beta_init = beta
    beta_mat[, i + 1] = beta
    
    # Decrease step size a bit
    #alpha = alpha * 0.9
  }
  
  # Return the matrix of x values, as well as the vector of function values across iterations, including the starting point (both have nIter + 1 elements, for beta_mat put them in columns)
  return(list(beta_mat = beta_mat, fvec = fvec))
}


#######################################
# Example
########################################

# Initialize response and covariates
########################################################################
y = c(1, 1, 0, 0, 1, 0, 1, 0, 0, 0) # response
n = length(y) # number of samples
x1 = c(8, 14, -7, 6, 5, 6, -5, 1, 0, -17) # covariates for 1st variable
X = cbind(rep(1, n), x1) # add a column of 1s for the intercept

# Apply vector form of steepest descent
#######################################################################

beta_init = c(0, 0.2) # initial starting value
alpha = 0.05 # step size
nIter = 500 # number of iterations

# Full Gradient Descent
out_full <- SteepestDescentBinLogistic(X, y, beta_init = beta_init, alpha = alpha, nIter = nIter, b = n)

plot(1:nIter, out_full$fvec[-1], type = 'o', xlab = "Iteration", ylab = "f(beta)", ylim = c(0.56, 0.63))

min(out_full$fvec) #0.57

# Standard Stochastic Gradient Descent
out_standard <- SteepestDescentBinLogistic(X, y, beta_init = beta_init, alpha = alpha, nIter = nIter, b = 1)

plot(1:nIter, out_standard$fvec[-1], type = 'o', xlab = "Iteration", ylab = "f(beta)", ylim = c(0.56, 0.63))

min(out_standard$fvec)


# Mini-batch stochastic gradient descent
out_batch <- SteepestDescentBinLogistic(X, y, beta_init = beta_init, alpha = alpha, nIter = nIter, b = 5)

plot(1:nIter, out_batch$fvec[-1], type = 'o', xlab = "Iteration", ylab = "f(beta)", ylim = c(0.56, 0.63))

min(out_batch$fvec)


# SAG stochastic gradient descent
out_sag <- SteepestDescentBinLogisticSAG(X, y, beta_init = beta_init, alpha = alpha, nIter = nIter)

plot(1:nIter, out_sag$fvec[-1], type = 'o', xlab = "Iteration", ylab = "f(beta)", ylim = c(0.56, 0.63))

min(out_sag$fvec)
