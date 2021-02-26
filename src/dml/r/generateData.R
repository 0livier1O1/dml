source("tools.r")
library(MASS) # for drawing from multivariate normal distribution

for(i in 1:1){
  data <- generate.data(k=90, n=100, dgp='linear')
  write.csv(data, paste0("data_", i,".csv", sep=""))
}
