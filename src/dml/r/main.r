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

args = commandArgs(trailingOnly=TRUE)
data.idx  <- 1 #as.integer(args[1])
iteration <- 1 #as.integer(args[2])
niter <- 7

print(paste0("Reading Data File", data.idx))
# data <- read.csv(paste0('data_', data.idx, '.csv'), row.names = 1)
data <- generate.data(k=500, n=1000, dgp='nonlinear')

y <- c("y.PLR")
d <- c("gamma.PLR")

methods <- c("Tree")

input.y <- as.matrix(data[, y])
input.d <- as.matrix(data[, d])
input.x <- data[, !colnames(data) %in% c(y, d)]

cat(input.y[1:10])

colnames(input.y) <- y
colnames(input.d) <- d
tic()
results.dml <- mcdml(y = input.y, d = input.d, x = input.x, methods=methods, data.idx=data.idx, iter.idx=iteration, niterations=niter)
cat('\n')
toc()
