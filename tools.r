
error <- function(y.pred, y){
  # Function to compute the root mean squarred error given a prediction and a truth
  error <- sqrt(mean((y.pred-y)^2))
  mis <- NA
  
  return(list(rmse = error, misrate=mis));
}

is.binary <- function(y){
  # Function to check if a vector y only contains 0 and 1
  y <- factor(y)
  binary = all(levels(y) %in% c("0", "1"))
  return(binary)
}