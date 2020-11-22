# Monte Carlo estimation of DML models

require("momentEstimation.r")

mcdml <- function(y, d, x, niterations, methods){
  # x is covariates
  # y is dependent variable
  # d is the treatment covariate
  # methods are ML methods to use
  
  nfold = 2
  
  ################################ Inputs ########################################
  # Forest <- list(clas_nodesize=1, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
  
  ################################ Estimation ####################################
  r <- foreach(k = 1:niterations, .combine='rbind', .inorder=FALSE, .packages=c('MASS','randomForest')) %dopar% { 
    dml.result <- dml(y, methods=methods, nfold=nfold)
  }
  
}