# Main file for generating MC simulation 
# Thanks are due to Nadja Van't Hoff for helping with getting me set up 
# I merely refactored the code available at https://github.com/demirermert/MLInference for my own needs 

rm(list = ls())
options(warn=-1)

library(MASS) # for drawing from multivariate normal distribution
library(scales)

source("MLestimators.r")
source("MonteCarloDML.r")
source("MomentEstimation.r")  

####################################################################################################
########################################### linear case ############################################
####################################################################################################

# #### Generate the data ####
# 
# theta <- 2 # coefficient of interest
# iter <- 2 # number of splits inside the DML
# k <- 20 # number of explanatory var
# n <- 100 # sample size
# 
# # define errors
# error.PLR <- mvrnorm(n, mu=c(0,0), Sigma=diag(2))
# 
# # Generate explanatory variables
# Sigma <- 0.2^t(sapply(1:k, function(i, j) abs(i-j), 1:k)) # Covariance structure
# x <- mvrnorm(n, mu=rep(0,k), Sigma = Sigma)
# colnames(x) <- paste("x", 1:k, sep="")
# 
# # define data for g and m
# x.main <- x[,1:10]
# x.confound <- x[,7:13] # four variables overlapping with y, seven in total
# 
# # PLR - Linear
# c.sample = c(-5, -3, -1, 1, 3, 5)
# c1 = sample(c.sample, NCOL(x.main), replace=TRUE)
# c2 = sample(c.sample, NCOL(x.confound), replace=TRUE)
# gamma.PLR <- rescale(x.confound %*% c2, to=c(-10,10)) + error.PLR[, 1]
# colnames(gamma.PLR) <- "gamma.PLR"
# 
# y.PLR <- gamma.PLR * theta + rescale(x.main %*% c1, to=c(-10,10)) + error.PLR[,2]
# colnames(y.PLR) <- "y.PLR"
# 
# 
# ### full data matrix
# data <- as.matrix(cbind(y.PLR, gamma.PLR, x))
# 
# y <- c("y.PLR")
# d <- c("gamma.PLR")
# 
# case = 1
# methods <- c("Elnet")
# 
# input.y <- as.matrix(data[, y[case]])
# input.d <- as.matrix(data[, d[case]])
# input.x <- x
# 
# colnames(input.y) <- y[case]
# colnames(input.d) <- d[case]
# 
# {
#   y = input.y
#   d = input.d
#   x = input.x
# }
# results.dml <- mcdml(y = input.y, d = input.d, x = input.x, niterations=iter, methods=methods)
# results.ols  <- summary(lm(input.y ~ input.d + input.x))$coefficients[2,1]

####################################################################################################
######################################## Nonlinear case ############################################
####################################################################################################

theta <- 2 # coefficient of interest
iter <- 100 # number of splits inside the DML
k <- 90 # number of explanatory var
n <- 100 # sample size

# Generate x
Sigma <- 0.2^t(sapply(1:k, function(i, j) abs(i-j), 1:k)) # Covariance structure
x <- mvrnorm(n, mu=rep(0,k), Sigma = Sigma)
colnames(x) <- paste("x", 1:k, sep="")


inv.exp <- function(x, k, c, d, p=0) {
  #out <- p+k/(1+exp(-c*(abs(x) - d)))
  out <- exp(x)
  return(out)
}

x.transform.g <- cbind(inv.exp(x[, 1], 2, 20, 1/2, 1), inv.exp(x[, 2], 2, 20, 1/3),
                       inv.exp(x[, 3], 2, 10, 1/3), inv.exp(x[, 4], 2, 12, 1/2, 1),
                       x[, 5] * x[, 6], x[, 7] * x[, 8], x[, 9]^2, x[, 10]^2,
                       log(abs(x[, 11] + 1)), log(abs(x[, 12] + 1)))


x.transform.m <- cbind(inv.exp(x[, 1], 2, 12, 1/2), inv.exp(x[, 3], 2, 12, 1/2),
                       x[, 14] * x[, 15], x[, 16] * x[, 17], x[, 11]^2, log(abs(x[, 9] + 1)),
                       log(abs(x[, 18] + 1)))

# Sample coefficients and transform the data non linearly
c.sample = c(-5, -3, -1, 1, 3, 5)
c1 = sample(c.sample, NCOL(x.transform.g), replace=TRUE)
c2 = sample(c.sample, NCOL(x.transform.m), replace=TRUE)


# Generate ys and gamma
error.PLR <- mvrnorm(n, mu=c(0,0), Sigma=diag(2))

gamma.PLR <- rescale(x.transform.m %*% c2, to=c(-10,10)) + error.PLR[, 1]
y.PLR     <- theta * gamma.PLR + rescale(x.transform.g %*% c1, to=c(-10,10)) + error.PLR[, 2]
# PLR - Linear
colnames(y.PLR) <- "y.PLR"
colnames(gamma.PLR) <- "gamma.PLR"

### full data matrix
data <- as.matrix(cbind(y.PLR, gamma.PLR, x))

y <- c("y.PLR")
d <- c("gamma.PLR")

methods <- c("Tree", "Forest", "Lasso", "Nnet", "Elnet", "Boosting")

input.y <- as.matrix(data[, y])
input.d <- as.matrix(data[, d])
input.x <- x

colnames(input.y) <- y
colnames(input.d) <- d


results.PLR <- matrix(NA, 1, length(methods) + 2)

results.dml <- mcdml(y = input.y, d = input.d, x = input.x, niterations=iter, methods=methods)
results.ols <- summary(lm(input.y ~ input.d + input.x))$coefficients[2,1]

results.PLR[, 1] <- as.numeric(results.ols)
results.PLR[, 2:(length(methods) + 2)] <- as.numeric(results.dml[1, ])

write.csv(results.PLR, paste0("PLR_", sample(1:1000000, 1),".csv", sep=""))
results.dml
results.ols
