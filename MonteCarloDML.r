# Monte Carlo estimation of DML models
library(MASS)
library(rpart)
library(sandwich)

source("momentEstimation.r")
source("MLestimators.r")

mcdml <- function(y, d, x, niterations, methods){
  # x is covariates
  # y is dependent variable
  # d is the treatment covariate
  # methods are ML methods to use
  
  M = length(methods)
  
  nfold <- 2
  
  data <- as.matrix(cbind(y, d, x))
  y <- colnames(y)
  d <- colnames(d)
  
  ################################ Inputs ########################################
  Forest <- list(clas_nodesize=1, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
  Tree <- list(reg_method="anova", clas_method="class", control = rpart.control(cp = 0.05)) # TODO FIX ARGUMENTS TREES
  
  ml.settings <- list(Tree=Tree, Forest=Forest)
  
  
  ################################ MC Estimation ####################################
  package_used <- c('MASS', 'sandwich', 'rpart')
  
  r <- foreach(k = 1:niterations, .combine='rbind', .inorder=FALSE, .packages=package_used) %dopar% { 
    dml.result <- dml(data, y, d, nfold, methods=methods, ml.settings=ml.settings, small_sample_DML = FALSE, model="plinear")
    data.frame(t(dml.result[1,]), t(dml.result[2,]))
  }
  
  r <- as.matrix(r)

  ################################ Compute and Format Output ##############################################
  
  result           <- matrix(0, 4, M+1)
  colnames(result) <- cbind(t(methods), "best")
  rownames(result) <- cbind("Mean ATE", "Median ATE", "se(median)",  "se")
  
  result[1,]        <- colMeans(r[, 1:(M+1)])
  result[2,]        <- colQuantiles(r[,1:(M+1)], probs=0.5) # met colquantiles bepaal je de median
  result[3,]        <- colQuantiles(sqrt(r[, (M+2):ncol(r)]^2 + (r[, 1:(M+1)] - colQuantiles(r[,1:(M+1)], probs=0.5))^2), probs=0.5)
  result[4,]        <- colQuantiles(r[,(M+2):ncol(r)], probs=0.5)
  
  result_table <- round(result, digits = 10)
  
  return(result_table)
}
