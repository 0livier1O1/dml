###############  ML Estimator: DECISION TREES ################### 

tree <- function(main, aux, formula, args){
  # main is the main subsample
  # aux is the auxiliary subsample 
  set.seed(5)
  tree            <- do.call(rpart, append(list(formula=formula, data=main), args))  # estimate with rpart (trees)
  complexityParam <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
  prunedTree      <- prune(tree, cp=complexityParam)
  
  # Compute residuals on main sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=main); 
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
  set.seed(5)
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
