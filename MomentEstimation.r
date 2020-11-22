dml <- function(data, y, nfold, methods){
  # This function estimates dml 
  n <- nrwo(y)
  M <- length(methods)
  
  if (nfold == 1) {
    cv.group <- rep(1, n)
  } else {
    split      <- runif(n)
    cv.group   <- as.numeric(cut(split, quantile(split, probs = seq(0, 1, 1/nfold)), include.lowest = TRUE))  
  }
  
  for (m in 1:M) { # Iterate over all the methods 
    
    # Cross validation groups (TO DO: Can be improved -> Use package)
    for (gr in 1:nfold){
      obs.main <- cvgroup == gr
      obs.aux <- cvgroup != gr
      
      sample.main <- as.data.frame(data[obs.main, ])
      sample.aux  <- as.data.frame(data[obs.aux, ])
    }
    
  }
  
}