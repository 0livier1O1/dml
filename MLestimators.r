###############  ML Estimators: Trees ################### 

tree <- function(mainSample, auxSample, formula, args){
  
  tree            <- do.call(rpart, append(list(formula=form, data=mainSample), args))  # estimate with rpart (trees)
  complexityParam <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]
  prunedTree      <- prune(trees,cp=bestcp)
  
  # Compute residuals on main sample
  linearModel    <- lm(formula,  x = TRUE, y = TRUE, data=mainSample); 
  yhat.main      <- predict(prunedTree, newdata=mainSample)
  resid.main     <- linearModel$y - yhat.main
  x.main         <- linearModel$x
  
  # Compute residuals on auxiliary sample
  linearModel    <- lm(form,  x = TRUE, y = TRUE, data=auxSample); 
  yhat.aux       <- predict(prunedTree, newdata=auxSample)
  resid.aux      <- linearModel$y - yhat.aux
  x.aux          <- linearModel$x
  
  return(list(yhat.main = yhat.main, 
              resid.main=resid.main, 
              yhat.aux  = yhat.aux, 
              resid.aux=resid.aux, 
              x.main=x.main, 
              x.aux=x.aux, model=prunedTree));
}