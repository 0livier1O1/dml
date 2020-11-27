# Main file for generating MC simulation 
# Thanks are due to Nadja Van't Hoff for providing her code on which 95% of this program is based, 
# I merely refactored, improved and extended the code to include non-linear cases. 


setwd("~/Documents/Rotterdam University/RA/dml/r")
rm(list = ls())

library(MASS) # for drawing from multivariate normal distribution
library(scales)

source("MLestimators.r")
source("MonteCarloDML.r")
source("momentEstimation.R")  

#### Generate the data ####

# Coefficients
c1 <- c(-3, -5, -1, -3, -1, 1, 3, 1, -1, 3) # 10 for g
c2 <- c(-1, -3, 1, -1, 3, 3, 5) # 7 for m

theta <- 2 # coefficient of interest
iter <- 10 # number of iterations in the Monte Carlo simulations
k <- 20 # number of explanatory var
n <- 100 # sample size

# define errors
set.seed(5)
error.PLR <- mvrnorm(n, mu=c(0,0), Sigma=diag(2))

# Generate explanatory variables
Sigma <- 0.2^t(sapply(1:k, function(i, j) abs(i-j), 1:k)) # Covariance structure
x <- mvrnorm(n, mu=rep(0,k), Sigma = Sigma)
colnames(x) <- paste("x", 1:k, sep="")
x

# define data for g and m
x.main <- x[,1:10]
x.confound <- x[,7:13] # four variables overlapping with y, seven in total

# PLR - Linear
gamma.PLR <- rescale(x.confound %*% c2, to=c(-10,10)) + error.PLR[, 1]
colnames(gamma.PLR) <- "gamma.PLR"

y.PLR <- gamma.PLR * theta + rescale(x.main %*% c1, to=c(-10,10)) + error.PLR[,2]
colnames(y.PLR) <- "y.PLR"


### full data matrix 
data <- as.matrix(cbind(y.PLR, gamma.PLR, x))

y <- c("y.PLR")
d <- c("gamma.PLR")

case = 1
methods <- c("Tree", "Forest")

input.y <- as.matrix(data[, y[case]])
input.d <- as.matrix(data[, d[case]])
input.x <- x

colnames(input.y) <- y[case]
colnames(input.d) <- d[case]

{
  y = input.y
  d = input.d
  x = input.x
}
results <- mcdml(y = input.y, d = input.d, x = input.x, niterations=iter, methods=methods)

