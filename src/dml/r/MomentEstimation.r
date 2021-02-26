source("tools.r")
source("MLestimators.r")

dml <- function(data, y, d, nfold, methods, ml.settings, small_sample_DML = FALSE, model="plinear"){
  # small_sample_DML uses DML2 algorithm instead
  
  # This function estimates dml 
  n <- nrow(data)
  M <- length(methods)
  
  # Initialise some variables
  mlestimates <- matrix(list(), length(methods), nfold)
  MSE.y       <- matrix(0, M + 1, nfold) # MSE for main equation 
  MSE.d       <- matrix(0, M + 1, nfold) # MSE for confounding equation
  TE          <- matrix(0, 1, M + 1) # Treatment effect DML
  STE         <- matrix(0, 1, M + 1) # Standard errors of DML
  
  y.resid.pooled <- vector("list", M + 1) # stacked residuals across sample splits
  d.resid.pooled <- vector("list", M + 1) # stacked residuals across sample splits
  
  # Define cross validation groups
  cv.group <- make.folds(nfold, n)
  # Moment estimation for each method
  for(m in 1:M) { # Iterate over all the methods 
    
    # Cross validaty.ion groups  
    # TODO: Can be improved -> Use package
    
    for(f in 1:nfold){
      obs.main <- cv.group != f 
      obs.aux <- cv.group == f
      
      sample.main <- as.data.frame(data[obs.main, ])
      sample.aux  <- as.data.frame(data[obs.aux, ])
      
      # Estimate ATE in the PLR model
      if (model == "plinear") {
        # TODO: This is a bad way to implement ensemble
        if (methods[m] %in% c("Ensemble", "ensemble")) {
          mlestimates[[m, f]] <- ensemble.estim(main=sample.main, aux=sample.aux, y, d, ml.settings=ml.settings)
        } else {
          mlestimates[[m, f]] <- mlestim(main=sample.main, aux=sample.aux, y, d, method=methods[m], ml.settings=ml.settings)  
        }

        MSE.y[m, f]              <- mlestimates[[m, f]]$y.error
        MSE.d[m, f]              <- mlestimates[[m, f]]$d.error
        
        # model residuals for moment equation estimation
        yresid = mlestimates[[m, f]]$y.resid
        dresid = mlestimates[[m, f]]$d.resid
        
        lm.fit.yresid       <- lm(as.matrix(yresid) ~ as.matrix(dresid) - 1); # Score function estimation
        
        ate                 <- lm.fit.yresid$coef; 
        HCV.coefs           <- vcovHC(lm.fit.yresid, type = 'HC') # Heteroskedasticity consistent covariance matrix
        STE[1, m]           <- ( 1/(nfold^2) ) * (diag(HCV.coefs)) + STE[1, m]  # DML1 -> added to the average over the folds
        TE[1, m]            <- ate/nfold  + TE[1, m] # DML1
        
        # Gather the residuals for DML2 
        
        y.resid.pooled[[m]]             <- c(y.resid.pooled[[m]], yresid)
        d.resid.pooled[[m]]             <- c(d.resid.pooled[[m]], dresid)
        

        # MSE1[(length(methods)+1),j] <- error(rep(mean(datause[,y], na.rm = TRUE), length(dataout[!is.na(dataout[,y]),y])), dataout[!is.na(dataout[,y]),y])$err
        # MSE2[(length(methods)+1),j] <- error(rep(mean(datause[,d], na.rm = TRUE), length(dataout[!is.na(dataout[,d]),d])), dataout[!is.na(dataout[,d]),d])$err
        
      }
    } # end folds iteration
    
  } # end methods iteration
  
  
  # Identify best performing methods for both equation of the "plinear" (partially linear) model  
  if(model=="plinear"){
    
    if(M>1){
      min1 <- which.min(rowMeans(MSE.y[1:M, ]))
      min2 <- which.min(rowMeans(MSE.d[1:M, ]))
    }
    else if(M==1){
      min1 <- which.min(mean(MSE.y[1, ]))
      min2 <- which.min(mean(MSE.d[1, ]))
      
    }
  }

  # Re estimate the main quantities with the best methods
  for (f in 1:nfold){
    if (model=="plinear"){
      yresid = mlestimates[[min1, f]]$y.resid
      dresid = mlestimates[[min2, f]]$d.resid
      
      lm.fit.yresid       <- lm(as.matrix(yresid) ~ as.matrix(dresid) - 1); # Score function estimation
      ate                 <- lm.fit.yresid$coef;
      HCV.coefs           <- vcovHC(lm.fit.yresid, type = 'HC') # Heteroskedasticity consistent covariance matrix
      STE[1, M+1]         <- ( 1/(nfold^2) ) * (diag(HCV.coefs)) + STE[1, M + 1]  
      TE[1, M+1]          <- ate/nfold + TE[1, M + 1] 
      
      y.resid.pooled[[M + 1]]  <- c(y.resid.pooled[[M + 1]], yresid) # This corresponds to the best 
      d.resid.pooled[[M + 1]]  <- c(d.resid.pooled[[M + 1]], dresid)
      
    }
  }
  
  # DML2 estimation
  # TODO REMOVE THE LOOP, JUST USE POOLED as matrix, not vectors
  if (small_sample_DML) {
    for(m in 1:(M+1)) {
      
      if(model=="plinear"){
        yresid = y.resid.pooled[[m]]
        dresid = d.resid.pooled[[m]]
        
        lm.fit.yresid     <- lm(as.matrix(yresid) ~ as.matrix(dresid) - 1); # Score function estimation
        ate               <- lm.fit.yresid$coef;
        HCV.coefs         <- vcovHC(lm.fit.yresid, type = 'HC');
        STE[1, m]         <- (diag(HCV.coefs))
        TE[1, m]          <- ate
      }
    }
  }
  # Prepare results
  results    <- matrix(0, 2, (length(methods)+1)) # (ATE + SE) x M matrix
  colnames(results)  <- c(methods, "best") 
  
  rownames(MSE.y)     <- c(methods, "best") 
  rownames(MSE.d)     <- c(methods, "best") 
  rownames(results)   <- c("ATE", "se")
  
  
  results[1,]         <- colMeans(TE)
  results[2,]         <- sqrt((STE))
  
  table <- rbind(results, rowMeans(MSE.y), rowMeans(MSE.d)) 
  rownames(table)[3:4]   <- c("MSE[Y|X]", "MSE[D|X]") 
  
  return(table)
  
}

mlestim <- function(main, aux, y, d, method, ml.settings) {
  # main is the main subsample
  # aux is the auxiliary subsample
  # method is the ML estimation method
  # settings are the parameters of the ML methods
  
  binary.d = is.binary(main[, d]) && is.binary(aux[, d])
  
  # formula for all covariates
  covariates <- colnames(main)
  covariates <- covariates[! covariates %in% c(y, d) ]
  formula.x <- paste(covariates, collapse="+")
  
  # TODO: Investigate this
  if (method %in% c("Lasso", "Ridge", "RLasso", "PostRLasso", "Elnet")){
    formula.x <- paste("(", formula.x , ")^2", sep="")
  }
  
  args = ml.settings[[method]]
  
  # select ML estimating method
  # ml.estimator will be a different function depending on the method
  
  if (method=="Tree") {                                    # Decision Tree
    args[which(names(args) %in% c("reg_method","clas_method"))] <-  NULL
    ml.estimator <- function(...) tree(...)
  } 
  else if (method %in% c("Forest", "TForest")) {           # (tuned) Random Forest
    tune = method == "TForest"
    
    ml.estimator <- function(...) RF(..., tune=tune) 
  } 
  else if (method == "Nnet"){                              # Neural Nets
    ml.estimator <- function(...) nnetF(...)
  } 
  else if (method %in% c("Ridge", "Lasso", "Elnet")){      # Penalised Regression
    alphas = list(Ridge=0, Lasso=1, Elnet=0.5)
    alpha = alphas[method]
    ml.estimator <- function(...) lassoF(..., alpha)
  }
  else if (method == "Boosting"){                          # Boosting
    distribution = args[["reg_dist"]]
    args[which(names(args) %in% c("clas_dist","reg_dist"))] <- NULL
    ml.estimator <- function(...) boost(..., distribution)
  }
  
  # Generate formula as input to ML estimation methods
  formula.y <- as.formula(paste(y, "~", formula.x))
  formula.d <- as.formula(paste(d, "~", formula.x)) 
  
  
  ## PROCEED WITH ML ESTIMATION ##
  estimator.fit.y <- ml.estimator(main=main, aux=aux, formula=formula.y, args=args)
  estimator.fit.d <- ml.estimator(main=main, aux=aux, formula=formula.d, args=args)
  
  ## TODO Add binary option for z
  
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

ensemble.estim <- function(main, aux, y, d, ml.settings){
  
  methods = ml.settings[["Ensemble"]]
  nfold <- 2  
  M <- length(methods)
  n <- nrow(main)
  
  mlestimates <- vector("list", M) # to store estimation results
  
  # Allocate to cross validation group
  cv.group   <- cv.group <- make.folds(nfold, n)
  
  # Create set of weights for each method
  # TODO: Improve this, might be slow for large k
  lst      <- lapply(numeric(M), function(x) as.vector(seq(0, 1, 0.01)))
  weights  <- as.matrix(expand.grid(lst))
  weights  <- weights[rowSums(weights)==1, ]
  
  
  outsample.pred <- array(0, dim = c(nrow(aux), M, 2))  
  errorM         <- array(0, dim = c(nrow(weights), nfold, 2))
  
  for(f in 1:nfold){
    obs.main <- cv.group == f
    obs.aux <- cv.group != f
    
    in.sample.main <- as.data.frame(main[obs.main, ])
    in.sample.aux  <- as.data.frame(main[obs.aux, ])
    
    in.sample.pred    <- array(0, dim=c(nrow(in.sample.aux), M, 2))  
    
    for(m in 1:M) { # Iterate over the methods
      
      mlestimates[[m]] <- mlestim(main=in.sample.main, aux=in.sample.aux, y, d, method=methods[m], ml.settings=ml.settings)
      
      in.sample.pred[, m, 1] <- mlestimates[[m]]$y.pred 
      in.sample.pred[, m, 2] <- mlestimates[[m]]$d.pred 
    }
    
    ensemble.y <- weights %*% t(in.sample.pred[, , 1]) # Weight methods to create ensemble estimates for each y and D
    ensemble.d <- weights %*% t(in.sample.pred[, , 2])
    
    errorM[, f, 1] = apply(ensemble.y, 1, function(x) error(x, in.sample.aux[, y])$rmse) # For each set of weight, compute rmse
    errorM[, f, 2] = apply(ensemble.d, 1, function(x) error(x, in.sample.aux[, d])$rmse)
    
  }
  weights.y <- which.min(as.matrix(rowSums(errorM[, , 1])))
  weights.d <- which.min(as.matrix(rowSums(errorM[, , 2])))
  
  # estimate again with best weights
  for(m in 1:M){
    mlestimates[[m]] <- mlestim(main=main, aux=aux, y, d, method=methods[m], ml.settings=ml.settings)
    
    outsample.pred[, m, 1] <- mlestimates[[m]]$y.pred 
    outsample.pred[, m, 2] <- mlestimates[[m]]$d.pred 
  }
  
  y.pred   <- outsample.pred[, , 1] %*% (weights[weights.y,])
  y.resid  <- aux[, y] - y.pred
  y.error  <- error(y.pred, aux[, y])$rmse
  
  d.pred   <- outsample.pred[, , 2] %*% (weights[weights.d,])
  d.resid  <- aux[, d] - d.pred
  d.error  <- error(d.pred, aux[, d])$rmse
  
  return(list(y.pred=y.pred,
              y.resid=y.resid,
              y.error=y.error,
              d.pred=d.pred,
              d.error=d.error,
              d.resid=d.resid))
}


