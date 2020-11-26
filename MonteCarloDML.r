# Monte Carlo estimation of DML models
library(MASS)
library(rpart)
library(sandwich);

source("momentEstimation.r")
source("MLestimators.r")

mcdml <- function(y, d, x, niterations, methods){
  # x is covariates
  # y is dependent variable
  # d is the treatment covariate
  # methods are ML methods to use
  
  nfold <- 2
  
  data <- as.matrix(cbind(y, d, x))
  y <- colnames(y)
  d <- colnames(d)
  
  ################################ Inputs ########################################
  Forest <- list(clas_nodesize=1, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
  Tree <- list(reg_method="anova", clas_method="class", control = rpart.control(cp = 0.05)) # TODO FIX ARGUMENTS TREES
  
  ml.settings <- list(Tree=Tree, Forest=Forest)
  ################################ Estimation ####################################
  #r <- foreach(k = 1:niterations, .combine='rbind', .inorder=FALSE, .packages=c('MASS','randomForest')) %dopar% { 
  #  dml.result <- dml(data, y, d, methods=methods, nfold=nfold)
  #}
  model = "plinear"
  dml.result <- dml(data, y, d, nfold, methods=methods, ml.settings=ml.settings, small_sample_DML = FALSE, model="plinear")
  # This one single run -> Use monte carlo
  return(dml.result)
}
