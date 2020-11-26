###############  ML Estimators: Trees ################### 

tree <- function(main, aux, formula, args){
  # main is the main subsample
  # aux is the auxiliary subsample 
  
  tree            <- do.call(rpart, append(list(formula=formula, data=main), args))  # estimate with rpart (trees)
  complexityParam <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
  prunedTree      <- prune(tree, cp=complexityParam)
  
  # Compute residuals on main sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=main); 
  yhat.main      <- predict(prunedTree, newdata=main)
  resid.main     <- linearModel$y - yhat.main
  x.main         <- linearModel$x
  
  # Compute residuals on auxiliary sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=aux); 
  yhat.aux       <- predict(prunedTree, newdata=aux)
  resid.aux      <- linearModel$y - yhat.aux
  x.aux          <- linearModel$x
  
  return(list(yhat.main = yhat.main, 
              resid.main=resid.main, 
              yhat.aux=yhat.aux, 
              resid.aux=resid.aux, 
              x.main=x.main, 
              x.aux=x.aux, model=prunedTree));
}

RF <- function(main, aux, formula, args){
  print("RF not defined yet")
  stop()
}

