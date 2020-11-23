source("tools.r")
source("MLestimators.r")

dml <- function(data, y, d, nfold, methods, ml.settings, model="plinear"){
  # This function estimates dml 
  n <- nrow(data)
  M <- length(methods)
  
  # Initialise some variables
  mlestimates <- matrix(list(), length(methods), nfold)
  MSE.y       <- matrix(0, M + 1, nfold) # MSE for main equation 
  MSE.d       <- matrix(0, M + 1, nfold) # MSE for confounding equation
  TE          <- matrix(0, 1, M + 1) ## TODO : WHAT IS THIS ? 
  STE         <- matrix(0, 1, M + 1) ## TODO : WHAT IS THIS ? 
  
  if (nfold == 1) {
    cv.group <- rep(1, n)
  } else {
    split      <- runif(n)
    cv.group   <- as.numeric(cut(split, quantile(split, probs = seq(0, 1, 1/nfold)), include.lowest = TRUE))  
  }
  
  # Moment estimation for each method
  for (m in 1:M) { # Iterate over all the methods 
    
    # Cross validation groups  
    # TODO: Can be improved -> Use package
    for (f in 1:nfold){
      obs.main <- cv.group == f
      obs.aux <- cv.group != f
      
      sample.main <- as.data.frame(data[obs.main, ])
      sample.aux  <- as.data.frame(data[obs.aux, ])
      
      if (model == "plinear") {
        mlestimates[[m, f]] <- mlestim(main=sample.main, aux=sample.aux, y, d, method=methods[m], ml.settings=ml.settings)
        
        MSE.y[m, f]              <- mlestimates[[m, f]]$y.error
        MSE.d[m, f]              <- mlestimates[[m, f]]$d.error
        
        # model residuals for moment equation estimation
        lm.fit.yresid       <- lm(as.matrix(mlestimates[[m, f]]$y.resid) ~ as.matrix(mlestimates[[m, f]]$d.resid) - 1);
        ate                 <- lm.fit.yresid$coef;
        HCV.coefs           <- vcovHC(lm.fit.yresid, type = 'HC') # Heteroskedasticity consistent covariance matrix
        STE[1, f]           <- ( 1/(nfold^2) ) * (diag(HCV.coefs)) + STE[1, f] 
        TE[1, f]            <- ate/nfold + TE[1, f]
        
        # TODO: FIX THIS 
        #ypool[[k]]             <- c(ypool[[k]], cond.comp[[k,j]]$ry)
        #zpool[[k]]             <- c(zpool[[k]], cond.comp[[k,j]]$rz)
        
        
        # MSE1[(length(methods)+1),j] <- error(rep(mean(datause[,y], na.rm = TRUE), length(dataout[!is.na(dataout[,y]),y])), dataout[!is.na(dataout[,y]),y])$err
        # MSE2[(length(methods)+1),j] <- error(rep(mean(datause[,d], na.rm = TRUE), length(dataout[!is.na(dataout[,d]),d])), dataout[!is.na(dataout[,d]),d])$err
        
      }
    } # end folds iteration
    
    # Identify best performing methods for both equation of the "plinear" (partially linear) model
    
  } # end methods iteration
  
  if(model=="plinear"){
    
    if(M>1){
      min1 <- which.min(rowMeans(MSE1[1:M, ]))
      min2 <- which.min(rowMeans(MSE2[1:M, ]))
    }
    else if(M==1){
      min1 <- which.min(mean(MSE1[1, ]))
      min2 <- which.min(mean(MSE2[1, ]))
      
    }
  }
  
  # Re estimate the main quantities with the best methods
  
  for (f in 1:nfold){
    if (model=="plinear"){
      lm.fit.yresid       <- lm(as.matrix(mlestimates[[min1, f]]$y.resid) ~ as.matrix(mlestimates[[min2, f]]$d.resid) - 1);
      ate                 <- lm.fit.yresid$coef;
      HCV.coefs           <- vcovHC(lm.fit.yresid, type = 'HC') # Heteroskedasticity consistent covariance matrix
      STE[1, M+1]           <- ( 1/(nfold^2) ) * (diag(HCV.coefs)) + STE[1, M + 1]  
      TE[1, M+1]            <- ate/nfold + TE[1, M + 1] # TODO why are STE AND TE ADDED TO THE PREVIOUS ONE 
    }
  }
  
}

mlestim <- function(main, aux, y, d, x, method, ml.settings) {
  # main is the main subsample
  # aux is the auxiliary subsample
  # method is the ML estimation method
  # settings are the parameters of the ML methods
  
  binary.d = is.binary(main[, d]) && is.binary(aux[, d])
  
  # formula for all covariates
  covariates <- colnames(main)
  covariates <- covariates[! covariates %in% c(y, d) ]
  formula.x <- paste(covariates, collapse="+")
  
  args = ml.settings[[method]]
  
  # select ML estimating method
  if (method=="Tree") {
    args[which(names(args) %in% c("reg_method","clas_method"))] <-  NULL
    ml.estimator <- function(...) tree(...)
  } else if (method %in% c("Forest", "TForest")) {
    tune = method == "TForest"
    
    ml.estimator <- function(...) RF(..., tune=tune) 
  } else if (method == "Lasso"){
    # for linear models formula
    formuma.x <- paste("(", covariates , ")^2", sep="")
    
    ml.estimator <- function(...) print("Lasso not defined yet")
  }
  
  # Generate formula as input to ML estimation methods
  formula.y <- as.formula(paste(y, "~", formula.x))
  formula.d <- as.formula(paste(d, "~", formula.x))
  
  
  ## PROCEED WITH ESTIMATION ##
  {  ## TODO PLAYGROUNDDD
    formula=formula.y
  }
  estimator.fit.y <- ml.estimator(main=main, aux=aux, formula=formula.y, args=args)
  estimator.fit.d <- ml.estimator(main=main, aux=aux, formula=formula.d, args=args)
  
  ## TO DO: if (binary.d) mis.z 
  
  y.pred  <- estimator.fit.y$yhat.aux
  y.resid <- estimator.fit.y$resid.aux
  y.error <- error(y.pred, aux[, y])$rmse
  
  d.pred   <- estimator.fit.d$yhat.aux
  d.error  <- error(d.pred, aux[, d])$rmse
  d.resid  <- estimator.fit.d$resid.aux
  
  return(list(y.pred=y.pred,
              y.resid=y.resid,
              y.error=y.error,
              d.pred=d.pred,
              d.error=d.error,
              d.resid=d.resid))
}


