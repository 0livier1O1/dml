###############  ML Estimator: DECISION TREES ################### 

tree <- function(main, aux, formula, args){
  # main is the main subsample
  # aux is the auxiliary subsample 
  trees           <- do.call(rpart, append(list(formula=formula, data=main), args))  # estimate with rpart (trees)
  complexityParam <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]
  prunedTree      <- prune(trees, cp=complexityParam)
  
  # Compute residuals on main sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=main)
  yhat.main      <- predict(prunedTree, newdata=main)
  resid.main     <- linearModel$y - yhat.main
  
  # Compute residuals on auxiliary sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=aux); 
  yhat.aux       <- predict(prunedTree, newdata=aux)
  resid.aux      <- linearModel$y - yhat.aux
  
  return(list(yhat.main = yhat.main, 
              resid.main=resid.main, 
              yhat.aux=yhat.aux, 
              resid.aux=resid.aux, 
              model=prunedTree));
}

###############  ML Estimator: RANDOM FOREST ################### 
RF <- function(main, aux, formula, args, tune=FALSE){
  # main is the main subsample
  # aux is the auxiliary subsample 
  # TODO: Allow tuning
  forest <- do.call(randomForest, append(list(formula=formula, data=main), args))
  
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=main); 
  yhat.main      <- as.numeric(forest$predicted)
  resid.main     <- as.numeric(linearModel$y) -  yhat.main
  
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=aux);    
  yhat.aux       <- predict(forest, aux, type="response")
  resid.aux      <- linearModel$y - as.numeric(yhat.aux)
  
  return(list(yhat.main = yhat.main, 
              resid.main=resid.main, 
              yhat.aux=yhat.aux, 
              resid.aux=resid.aux, 
              model=forest));
}

###############  ML Estimator: Neural Nets ################### 
nnetF <- function(main, aux, formula, args){
  # main is the main subsample
  # aux is the auxiliary subsample 
  # TODO WHY do we use min/maxs
  
  maxs <- apply(main, 2, max) 
  mins <- apply(main, 2, min)
  
  main[, ] <- as.data.frame(scale(main, center = mins, scale = maxs - mins))
  aux[, ] <- as.data.frame(scale(aux, center = mins, scale = maxs - mins))
  
  NN  <- do.call(nnet, append(list(formula=formula, data=main, linout=FALSE), args))  
  y <- toString(as.list(formula)[[2]]) # extract the y variable from the formula
  
  browser()
  
  # Compute residuals on main sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=main); 
  yhat.main      <- predict(NN, main)
  resid.main     <- (linearModel$y - yhat.main) * ((maxs[y]-mins[y]))
  
  # Compute residuals on auxiliary sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=aux); 
  yhat.aux       <- predict(NN, aux)
  resid.aux      <- (linearModel$y - yhat.aux) * ((maxs[y]-mins[y]))
  
  return(list(yhat.main = yhat.main, 
              resid.main=resid.main, 
              yhat.aux=yhat.aux, 
              resid.aux=resid.aux, 
              model=NN));
}

###############  ML Estimator: Lasso/Ridge/Elastic Nets ################### 
lassoF <- function(main, aux, formula, args, alpha){
  # main is the main subsample
  # aux is the auxiliary subsample 
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=main); 
  lasso          <- do.call(cv.glmnet, append(list(x=linearModel$x[ ,-1], y=linearModel$y, alpha=alpha), args))
  
  # Compute residuals on main sample
  yhat.main      <- predict(lasso, newx=linearModel$x[ ,-1])
  resid.main     <- linearModel$y - yhat.main
  
  # Compute residuals on auxiliary sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=aux)
  yhat.aux       <- predict(lasso, newx=linearModel$x[ ,-1])
  resid.aux      <- linearModel$y - yhat.aux
  
  return(list(yhat.main = yhat.main, 
              resid.main=resid.main, 
              yhat.aux=yhat.aux, 
              resid.aux=resid.aux, 
              model=lasso));
}

###############  ML Estimator: Boosting ################### 
boost <- function(main, aux, formula, args, distribution, ntrees=100){
  # main is the main subsample
  # aux is the auxiliary subsample 
  boostfit       <- do.call(gbm, append(list(formula=formula, data=main, distribution=distribution), args))
  
  # Compute residuals on main sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=main); 
  yhat.main      <- predict(boostfit, n.trees=ntrees)
  resid.main     <- linearModel$y - yhat.main
  
  # Compute residuals on auxiliary sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=aux)
  yhat.aux       <- predict(boostfit, n.trees=ntrees, newdata=aux,  type="response")
  resid.aux      <- linearModel$y - yhat.aux
  
  return(list(yhat.main = yhat.main, 
              resid.main=resid.main, 
              yhat.aux=yhat.aux, 
              resid.aux=resid.aux, 
              model=boostfit));
}
