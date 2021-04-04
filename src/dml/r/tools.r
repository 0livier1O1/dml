make.folds <- function(nfold, nobs){
  if (nfold == 1) {
    cv.group <- rep(1, nobs)
  } else {
    split      <- runif(nobs)
    
    cv.group   <- as.numeric(cut(split, quantile(split, probs = seq(0, 1, 1/nfold)), include.lowest = TRUE))  
  }
  return(cv.group)
}

error <- function(y.pred, y){
  # Function to compute the root mean squarred error given a prediction and a truth
  error <- sqrt(mean((y.pred-y)^2))
  mis <- NA
  
  return(list(rmse = error, misrate=mis));
}

is.binary <- function(y){
  # Function to check if a vector y only contains 0 and 1
  y <- factor(y)
  binary = all(levels(y) %in% c("0", "1"))
  return(binary)
}

generate.data <- function(k, n, theta=2, dgp='linear'){
  # Generate Errors
  error.PLR <- mvrnorm(n, mu=c(0,0), Sigma=diag(2))
  
  # Generate explanatory variables
  Sigma <- 0.2^t(sapply(1:k, function(i, j) abs(i-j), 1:k)) # Covariance structure
  x <- mvrnorm(n, mu=rep(0,k), Sigma = Sigma)
  colnames(x) <- paste("x", 1:k, sep="")
  
  
  if (dgp == 'linear'){
    x.g <- x[,1:10]
    x.m <- x[,7:13] # four variables overlapping with y, seven in total 
  } else {
    inv.exp <- function(x, k, c, d, p=0) {
      # out <- p+k/(1+exp(-c*(abs(x) - d)))
      out <- exp(x)
      return(out)
    }
    x.g <- cbind(inv.exp(x[, 1], 2, 20, 1/2, 1) * inv.exp(x[, 2], 2, 20, 1/3),
                 inv.exp(x[, 3], 2, 10, 1/3) * inv.exp(x[, 4], 2, 12, 1/2, 1),
                 x[, 5] * x[, 6], x[, 7] * x[, 8], x[, 9]^2, x[, 10]^2,
                 log(abs(x[, 11] + 1)) * log(abs(x[, 12] + 1)),
                 log(abs(x[, 4] + 1)),
                 1/x[, 5])
    x.m <- cbind(inv.exp(x[, 1], 2, 12, 1/2) * inv.exp(x[, 3], 2, 12, 1/2),
                 x[, 14] * x[, 15], x[, 16] * x[, 17], x[, 11]^2,
                 log(abs(x[, 9] + 1)) * log(abs(x[, 18] + 1)),
                 1/x[, 17],
                 log(abs(x[, 3] + 1)))
  }
  
  # Sample coefficients and transform the data non linearly
  c.sample = c(-5, -3, -1, 1, 3, 5)
  c1 = sample(c.sample, NCOL(x.g), replace=TRUE)
  c2 = sample(c.sample, NCOL(x.m), replace=TRUE)
  
  
  # Generate ys and gamma
  error.PLR <- mvrnorm(n, mu=c(0,0), Sigma=diag(2))
  gamma.PLR <- rescale(x.m %*% c2, to=c(-10,10)) + error.PLR[, 1]
  y.PLR     <- theta * gamma.PLR + rescale(x.g %*% c1, to=c(-10,10)) + error.PLR[, 2]
  
  colnames(gamma.PLR) <- "gamma.PLR"
  colnames(y.PLR) <- "y.PLR"
  
  data <- as.matrix(cbind(y.PLR, gamma.PLR, x))
  
  return(data)
}
