library(MASS)
library(rpart)
library(sandwich)
library(foreach)
library(gbm)
library(glmnet)
library(randomForest)
library(nnet)
library(matrixStats)
library(snow)
library(doParallel)

source("MomentEstimation.r")
source("MLestimators.r")
source("tools.r")


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
  
  ################################ ML Inputs ########################################
  Forest       <- list(clas_nodesize=1, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
  Tree         <- list(reg_method="anova", clas_method="class", control = rpart.control(cp = 0.05)) # TODO FIX ARGUMENTS TREES
  Nnet         <- list(size=8,  maxit=1000, decay=0.01, MaxNWts=10000,  trace=FALSE)
  Boosting     <- list(bag.fraction = .5, train.fraction = 1.0, interaction.depth=2, n.trees=1000, shrinkage=.01, n.cores=1, cv.folds=0, n.minobsinnode=2, verbose = FALSE, clas_dist= 'adaboost', reg_dist='gaussian')
  Lasso        <- list(nfolds = 10)
  ensemble     <- c("Forest", "Lasso", "Nnet", "Elnet")
  
  ml.settings <- list(Tree=Tree, Forest=Forest, Nnet=Nnet, Lasso=Lasso, Boosting=Boosting, Ensemble=ensemble)
  ################################ MC Estimation ####################################
  package_used <- c('MASS', 'sandwich', 'rpart', 'randomForest', 'gbm', 'glmnet', 'randomForest', 'nnet', 'matrixStats', 
                    'BBmisc')
  
  ################## Parallelization accross NODES (CPUs) ###########################
  
  #clusters <- getMPIcluster()
  browser()
  dml.result <- dml(data, y, d, nfold, methods=methods, ml.settings=ml.settings, small_sample_DML = TRUE, model="plinear")
  registerDoParallel(detectCores()-1)
  r <- foreach(k = 1:niterations, .combine='rbind', .inorder=FALSE, .packages=package_used) %dopar% { 
    dml.result <- dml(data, y, d, nfold, methods=methods, ml.settings=ml.settings, small_sample_DML = TRUE, model="plinear")
    data.frame(t(dml.result[1,]), t(dml.result[2,]), t(dml.result[3,]), t(dml.result[4,]))
  }
  r <- as.matrix(r)
  browser()
  
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
