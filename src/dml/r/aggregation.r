rm(list = ls())
options(warn=-1)

library(MASS) # for drawing from multivariate normal distribution
library(AER) # for IV regression
library(tictoc)
library(xtable)
library(matrixStats)

setwd("~/Documents/Rotterdam University/RA/dml/results 2 /")

# value of true parameter
theta <- 2
methods <- c("OLS", "Tree", "Forest", "Lasso", "Boosting", "Nnet", "Elnet", "Ensemble")

# load in files obtained from Lisa
k = 50
n = 1000
path1 = paste0('./k=', k, ', n=', n, '/')
path2 = paste0('k=', k, ', n=', n, '/')

temp = list.files(path=path1, pattern="*.csv")

results.PLR <- matrix(NA, length(temp), length(methods))

for (i in 1:length(temp)) {
  res <- read.csv(paste0(path2, temp[i]))
  
  results.PLR[i, ] <- as.matrix(res[1, 1:length(methods) + 1])
}

colnames(results.PLR) <- methods

##########################################################################################
### Evaluation of results                                                              ###
##########################################################################################

makeTable <- function(results) {
  
  result_table <- matrix(NA, 5, length(methods))
  colnames(result_table) <- methods
  rownames(result_table) <- c("Mean", "MAE", "Var", "MSE","OLS hitrate")
  
  # calculate mean, median, variance and MSE
  result_table[1,] <- colMeans(results)
  result_table[2,] <- colMedians(abs(theta - results))
  result_table[3,] <- colVars(results)
  result_table[4,] <- colMeans((theta - results)^2)
  
  # calculate hitrate
  rOLS <- abs(theta - results[, "OLS"])
  for (method in methods) {
    rMeth <- abs(theta - results[,method])
    result_table[5,method] <- sum(rOLS < rMeth) / length(rOLS)
  }
  
  return(t(result_table))
}
# save results
sim <- 20
print(xtable(makeTable(results.PLR), type="latex", digits=6),
      file=paste("2PLR_n", n, "_sim", sim, "_k", k,".txt",sep=""))

