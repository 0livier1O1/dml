
rootMSE <- function(y.pred, y){
  # Function to compute the root mean squarred error given a prediction and a truth
  err  <- sqrt(mean((y.pred-y)^2))
  
  return(rmse = err);
}