####### Function Definitions ####### 

#' Standardize a 1D vector
#' 
#' @param x vector of numeric values

standardize.1D <- function(x){
  if (sum(x) != 0){
    (x - min(x)) / (max(x) - min(x))
  }
  else x # in case x is a vector of zeroes
}



#' Function to simulate observations from Nicholson's blowfly model
#' 
#' @param delta
#' @param P
#' @param betaEpsilon
#' @param betaE
#' @param phi
#' @param constants list containing values of the constants for the blowfly model
#' @param n.reps number of sets of observations to simulate
#' @param standardize boolean, whether to standardize the output observations to 
#'                    values between 0 and 1

simulate_NBF <- function(delta, P, betaEpsilon, betaE, phi, 
                         constants=list(n=300, tau=14, N0=400), n.reps=1,
                         standardize=TRUE){
  
  # known parameters
  n <- constants$n
  tau <- constants$tau
  N0 <- constants$N0
  
  # initiate processes to simulate
  N <- R <- S <- y <- matrix(nrow=n, ncol=n.reps)
  
  # simulate error terms
  e <- epsilon <- matrix(nrow=n, ncol=n.reps)
  for (i in 1:n.reps){
    e[,i] = rgamma(n, betaE, rate=betaE)
    epsilon[,i] = rgamma(n, betaEpsilon, rate=betaEpsilon)
  }
  
  # initial survival process term
  S[1,] = rbinom(n.reps, N0, exp(-delta*epsilon[1,]))
  N[1,] = S[1,]
  
  # simulate survival and population processes up to time `tau`
  for(t in 2:(tau)){
    S[t,] = rbinom(n.reps, N[t-1,], exp(-delta*epsilon[t,]))
    N[t,] = S[t,]
  }
  
  # simulate survival, birth and population processes at time `tau`+1
  S[tau+1,] = rbinom(n.reps, N[tau,], exp(-delta*epsilon[tau+1,]))
  R[tau+1,] = rpois(n.reps, P*N0*e[tau+1,]*exp(-1))
  N[tau+1,] = S[tau+1,] + R[tau+1,]
  
  # simulate the rest of each of these processes
  for(t in (tau+2):n){
    R[t,] = rpois(n.reps, P*N[t-tau-1,]*e[t,]*exp(-N[t-tau-1,]/N0))
    S[t,] = rbinom(n.reps, N[t-1,], exp(-delta*epsilon[t,]))
    N[t,] = R[t,] + S[t,]
  }
  
  # observation process
  for(t in 1:n){
    y[t,] = rpois(n.reps, phi*N[t,])
  }
  
  # if required, standardize each column independently
  if (standardize){
    apply(y, 2, standardize.1D)
  } else {
    y
  }
}



#' Function for plotting bootstrap intervals
#' 
#' @param model the trained neural network 
#' @param pred point estimate, given by `model`, of the parameters
#' @param unknowns names of parameters being estimated
#' @param true_params true values of all five parameters
#' @param n_bootstraps number of bootstrap samples to generate
#' @param constants values of constants for the blowfly model
#' @param priors if not NULL, a dataframe containing the bounds of the prior 
#'               distributions for each parameter
#' @param log_scale boolean, whether to show all plots in log scale

bootstrap <- function(model, pred, unknowns, true_params, n_bootstraps, 
                      constants, priors=NULL, log_scale=FALSE){
  
  n <- constants$n
  k <- length(unknowns)
  param_names <- c('delta', 'P', 'betaEpsilon', 'betaE', 'phi')
  params <- true <- c()
  
  count <- 1
  for (i in seq_along(param_names)){
    if (!is.na(unknowns[count]) && param_names[i] == unknowns[count]){
      params[i] = pred[count]
      true[count] = true_params[[i]]
      count <- count + 1
    } else {
      params[i] = true_params[[i]]
    }
  }
  
  theta_bootstrap <- array(NA, c(k, n_bootstraps))
  y_bootstrap <- array(NA, c(n, n_bootstraps))
  lwr <- upr <- numeric(k)
  
  print('simulating bootstrap samples...')
  for (j in 1:n_bootstraps){
    y_bootstrap <- array(simulate_NBF(delta=params[1], P=params[2],
                                      betaEpsilon=params[3], betaE=params[4],
                                      phi=params[5], constants=constants,
                                      standardize=TRUE), c(1,n))
    
    if (log_scale) {
      theta_bootstrap[,j] <- model %>% predict(y_bootstrap, verbose=0)
    } else {
      theta_bootstrap[,j] <- exp(model %>% predict(y_bootstrap, verbose=0))
    }
  }
  
  for (i in 1:k){
    interval <- unname(quantile(theta_bootstrap[i,], probs = c(0.025, 0.975)))
    lwr[i] <- interval[1]
    upr[i] <- interval[2]
  }
  
  if (!is.null(priors)){
    
    if (log_scale){
      bootstrap_df <- merge(data.frame(param=unknowns, true=log(true), pred=log(c(pred)), 
                                       lwr=lwr, upr=upr), 
                            priors, by='param')
    } else {
      bootstrap_df <- merge(data.frame(param=unknowns, true=true, pred=c(pred), lwr=lwr, upr=upr), 
                            priors, by='param')
      bootstrap_df$prior_lwr = exp(bootstrap_df$prior_lwr)
      bootstrap_df$prior_upr = exp(bootstrap_df$prior_upr)
    }
    
    print(ggplot(bootstrap_df %>% group_by(param), aes(x=param)) +
            geom_point(aes(y=true, colour='true')) + 
            geom_point(aes(y=pred, colour='pred')) +
            geom_errorbar(aes(ymin=prior_lwr, ymax=prior_upr, colour='prior')) +
            geom_errorbar(aes(ymin=lwr, ymax=upr, colour='95% CI')) + 
            facet_wrap(~param, scales='free')
    )
  } else{
    
    if (log_scale){
      bootstrap_df <- data.frame(param=unknowns, true=log(true), pred=log(c(pred)), lwr=lwr, upr=upr)
    } else {
      bootstrap_df <- data.frame(param=unknowns, true=true, pred=c(pred), lwr=lwr, upr=upr)
    }
    
    print(ggplot(bootstrap_df %>% group_by(param), aes(x=param)) +
            geom_point(aes(y=true, colour='true')) + 
            geom_point(aes(y=pred, colour='pred')) +
            geom_errorbar(aes(ymin=lwr, ymax=upr, colour='95% CI')) + 
            facet_wrap(~param, scales='free')
    )
  }
}



#' Plots predictions from the network for samples generated by different parameter combinations.
#' Each parameter must have three associated test values, then samples are generated using every
#' possible combination of test values. This gives a total of three 9x9 plots.
#' 
#' @param model the trained neural network
#' @param delta vector of test values for parameter delta
#' @param P vector of test values for parameter P
#' @param phi vector of test values for parameter phi
#' @param true_params true values of all five parameters
#' @param constants values of constants for the blowfly model
#' @param n_samples number of samples to generate for each parameter combination
#' @param log_scale boolean, whether to show all plots in log scale

scatterplot3 <- function(model, delta, P, phi, true_params, constants, n_samples, 
                         log_scale=FALSE){
  
  betaEpsilon_true <- true_params$betaEpsilon
  betaE_true <- true_params$betaE
  n <- constants$n
  
  unknowns <- c('delta', 'P', 'phi')
  
  test_params <- list(data.frame(cbind(delta=expand.grid(delta, P)$Var1,
                                       P=expand.grid(delta, P)$Var2,
                                       phi=rep(phi_true, 9))),
                      
                      data.frame(cbind(delta=expand.grid(delta, phi)$Var1,
                                       P=rep(P_true, 9),
                                       phi=expand.grid(delta, phi)$Var2)),
                      
                      data.frame(cbind(delta=rep(delta_true, 9),
                                       P=expand.grid(P, phi)$Var1,
                                       phi=expand.grid(P, phi)$Var2)))
  
  for (df_num in 1:length(test_params)) {
    df <- test_params[[df_num]]
    df$comb <- c(1:9)
    
    i <- combn(3, 2)[,df_num][1]
    j <- combn(3, 2)[,df_num][2]
    
    comb_df <- data.frame(array(NA, c(n_samples*9, 3)))
    colnames(comb_df) <- c(unknowns[i], unknowns[j], 'comb')
    
    cat('simulating for', c('first', 'second', 'third')[df_num], 'set of plots... \n')
    
    for (comb in 1:9){
      param_comb <- df[comb,]
      
      for (sample_num in 1:n_samples){
        y_test[1,] <- array(simulate_NBF(delta=param_comb$delta, P=param_comb$P, betaEpsilon=betaEpsilon_true,
                                         betaE=betaE_true, phi=param_comb$phi, constants=constants,
                                         standardize=TRUE), c(1, n))
        
        if (log_scale){
          pred <- model %>% predict(y_test, verbose=0)
        } else {
          pred <- exp(model %>% predict(y_test, verbose=0))
        }
        comb_df[n_samples*(comb - 1) + sample_num,] = c(pred[i], pred[j], comb)
      }
    }
    
    if (log_scale){
      
      print(
        ggplot() + geom_point(comb_df, mapping=aes(x=!!sym(unknowns[i]),
                                                   y=!!sym(unknowns[j]))) + 
          geom_point(df, mapping=aes(x=log(!!sym(unknowns[i])),
                                     y=log(!!sym(unknowns[j])), colour='true'), shape=18, size=3) +
          facet_wrap(~comb) +
          xlab(colnames(comb_df)[1]) + ylab(colnames(comb_df)[2])
      )
      
    } else {
      
      print(
        ggplot() + geom_point(comb_df, mapping=aes(x=!!sym(unknowns[i]),
                                                   y=!!sym(unknowns[j]))) + 
          geom_point(df, mapping=aes(x=!!sym(unknowns[i]),
                                     y=!!sym(unknowns[j]), colour='true'), shape=18, size=3) +
          facet_wrap(~comb) +
          xlab(colnames(comb_df)[1]) + ylab(colnames(comb_df)[2])
      )
    }
  }
}



#' Function to obtain log synthetic likelihood for a sample `y`, given parameter 
#' estimates in `theta`
#' 
#' @param theta current estimate of the parameters delta, P and phi
#' @param constants values of constants for the blowfly model
#' @param true_betas true values of the two beta parameters
#' @param y observed blowfly sample for which to calculate the log SL
#' @param n.reps number of replicate data sets to simulate in order to obtain summary statistics
#' @param stats boolean, whether to return the summary statistics instead of the log SL

custom_sl <- function(theta, constants, true_betas, y, n.reps=500, stats=FALSE){
  
  delta <- theta[1]; P <- theta[2]; phi <- theta[3]
  betaEpsilon_true <- true_betas[1]; betaE_true <- true_betas[2]
  
  # simulate `n.reps` times from model with current parameter values
  Y <- simulate_NBF(delta=delta, P=P, betaEpsilon=betaEpsilon_true, betaE=betaE_true, 
                    phi=phi, constants=constants, n.reps=n.reps, standardize=FALSE)
  
  ## Now assemble the relevant statistics
  if (!is.matrix(y)) y <- matrix(y,length(y),1)
  
  acf.Y <- sl.acf(Y,max.lag=11)
  acf.y <- sl.acf(y,max.lag=11)
  
  b0.Y <- nlar(Y,lag=c(6,6,6,1,1),power=c(1,2,3,1,2))
  b0.y <- nlar(y,lag=c(6,6,6,1,1),power=c(1,2,3,1,2))
  
  b1.Y <- order.dist(Y,y,np=3,diff=1)
  b1.y <- order.dist(y,y,np=3,diff=1)   
  
  ## combine the statistics...
  
  sy <- c(as.numeric(acf.y),
          as.numeric(b0.y),
          as.numeric(b1.y),
          mean(y),
          mean(y)-median(y), ## possibly mean would be better here?
          sum(abs(diff(sign(diff(y)))))/2 ## count turning points
  )
  sY <- rbind(acf.Y,
              b0.Y, 
              b1.Y,
              colMeans(Y),
              colMeans(Y)-apply(Y,2,median),
              colSums(abs(diff(sign(diff(Y)))))/2
  )
  
  ## get the log synthetic likelihood
  sY <- sY[,is.finite(colSums(sY))]
  
  ##  sY <- trim.stat(sY) ## trimming the marginal extremes to robustify
  
  if (stats) { 
    attr(sY,"observed") <- sy 
    return(sY)   ## just return statistics
  }
  
  er <- robust.vcov(sY)
  
  ## robustify the likelihood...
  
  rss <- sum((er$E%*%(sy-er$mY))^2)
  
  ll0 <- -rss/2 - er$half.ldet.V ## true l_s
  
  d0 <- qchisq(.99,nrow(sY))^.5
  
  rss <- not.sq(sqrt(rss),alpha=.1,d0=d0)
  
  ll <- -rss/2 - er$half.ldet.V ## robustified l_s
  attr(ll,"true") <- ll0 ## append the true l_s
  ll
}






