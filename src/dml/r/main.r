# Main file for generating MC simulation 
# Thanks are due to Nadja Van't Hoff for helping with getting me set up 
# I merely refactored the code available at https://github.com/demirermert/MLInference for my own needs 

rm(list = ls())
options(warn=-1)

library(MASS) # for drawing from multivariate normal distribution
library(scales)
library(tictoc)
library(BBmisc)

source("tools.r")
source("MLestimators.r")
source("MonteCarloDML.r")
source("MomentEstimation.r")  

iter <- 100

data <- read.csv('~/uni/ra/dml/data/data_1.csv')

y <- c("y.PLR")
d <- c("gamma.PLR")

methods <- c("Tree", "Forest") #, "Nnet", "Elnet", "Boosting")

input.y <- as.matrix(data[, y])
input.d <- as.matrix(data[, d])
input.x <- data[, !colnames(data) %in% c(y, d)]

colnames(input.y) <- y
colnames(input.d) <- d

cat('Starting DML \n')
tic()
results.dml <- mcdml(y = input.y, d = input.d, x = input.x, niterations=iter, methods=methods)
toc()
cat('Finished DML \n')
results.dml

results.ols <- summary(lm(input.y ~ input.d + input.x))$coefficients[2,1]

results.PLR <- matrix(NA, 1, length(methods) + 2)
results.PLR[, 1] <- as.numeric(results.ols)
results.PLR[, 2:(length(methods) + 2)] <- as.numeric(results.dml[1, ])

# write.csv(results.PLR, paste0("PLR_", sample(1:1000000, 1),".csv", sep=""))
# results.ols
