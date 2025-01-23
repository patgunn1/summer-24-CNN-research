####### Function Definitions ####### 

#' Standardize a vector to values between 0 and 1
#' 
#' @param x vector of numeric values

standardize.1D <- function(x){
  if (sum(x) == 0){  # in case x is a vector of zeroes
    x
  }
  else{
    (x - min(x)) / (max(x) - min(x))
  }
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



#' Function for plotting 95% bootstrap intervals
#' 
#' @param model the trained neural network 
#' @param pred point estimate of the parameters given by `model`, in LOG scale
#' @param unknowns names of parameters being estimated
#' @param true_params true values of all five parameters, in REGULAR scale
#' @param n_bootstraps number of bootstrap samples to generate
#' @param constants values of constants for the blowfly model
#' @param priors if not NULL, a dataframe containing the LOG bounds of the 
#'               prior distributions for each parameter
#' @param MCMC_chains if not NULL, a matrix containing the MCMC chains for each 
#'                    parameter, in LOG scale
#' @param log_scale boolean, whether to show all plots in log scale

bootstrap <- function(model, pred, unknowns, true_params, n_bootstraps, 
                      constants, priors=NULL, MCMC_chains=NULL, log_scale=FALSE){
  
  n <- constants$n
  k <- length(unknowns)
  
  # colours for plots
  red <- '#F8766D'
  green <- '#7CAE00'
  blue <- '#00BFC4'
  purple <- '#C77CFF'
  black <- 'black'
  
  # this next block can be ignored, it allows the function to plot when estimating fewer than 5 params
  # but for 5 param estimation it does the exact same as setting `params <- pred`. note that `params`
  # will be in log scale!
  
  param_names <- c('delta', 'P', 'betaEpsilon', 'betaE', 'phi')
  params <- true <- c()
  count <- 1
  for (i in seq_along(param_names)){
    if (!is.na(unknowns[count]) && param_names[i] == unknowns[count]){
      params[i] = pred[count]
      true[count] = log(true_params[[i]])
      count <- count + 1
    } else {
      params[i] = log(true_params[[i]])
    }
  }
  
  print('simulating bootstrap samples...')
  
  # simulate bootstrap samples
  y_bootstrap <- t( # transpose is needed for compatibility with predict()
    simulate_NBF(delta=exp(params[1]), P=exp(params[2]), betaEpsilon=exp(params[3]), 
                 betaE=exp(params[4]), phi=exp(params[5]), constants=constants, 
                 n.reps=n_bootstraps, standardize=TRUE)
  )
  
  # obtain a prediction from CNN for each bootstrap sample
  theta_pred <- model %>% predict(y_bootstrap, verbose=0)
  
  # calculate middle 95% interval of the predictions for each param
  lwr <- upr <- numeric(k)
  for (i in 1:k){
    interval <- unname(quantile(theta_pred[,i], probs = c(0.025, 0.975)))
    lwr[i] <- interval[1]
    upr[i] <- interval[2]
  }
  
  ## prior bounds and MCMC results in each plot ##
  if (!is.null(priors) && !is.null(MCMC_chains)) {
    
    MCMC_quantiles <- hdi(MCMC_chains)  # 95% HDI of MCMC posteriors
    
    if (log_scale){
      
      MCMC_results <- data.frame(param=unknowns, MCMC.lwr=MCMC_quantiles$CI_low, 
                                 MCMC.upr=MCMC_quantiles$CI_high, 
                                 MCMC.mean=unname(apply(MCMC_chains, 2, mean)))
      
      bootstrap_df <- data.frame(param=unknowns, true=true, NN.pred=pred, 
                                 NN.lwr=lwr, NN.upr=upr) %>%
        merge(priors, by='param') %>%
        merge(MCMC_results, by='param')
      
    } else {  # not log scale
      
      MCMC_results <- data.frame(param=unknowns, MCMC.lwr=exp(MCMC_quantiles$CI_low), 
                                 MCMC.upr=exp(MCMC_quantiles$CI_high), 
                                 MCMC.mean=exp(unname(apply(MCMC_chains, 2, mean))))
      
      bootstrap_df <- data.frame(param=unknowns, true=exp(true), NN.pred=exp(pred), 
                                 NN.lwr=exp(lwr), NN.upr=exp(upr)) %>%
        merge(priors, by='param') %>%
        merge(MCMC_results, by='param')
      
      bootstrap_df$prior.lwr = exp(bootstrap_df$prior.lwr)
      bootstrap_df$prior.upr = exp(bootstrap_df$prior.upr)
    }
    
    
    print(ggplot(bootstrap_df %>% group_by(param), aes(x=param)) +
            geom_point(aes(y=true, colour='True'), size=3, shape=18) + 
            geom_errorbar(aes(ymin=prior.lwr, ymax=prior.upr, colour='Prior Bounds'), linewidth=0.8) +
            geom_point(aes(y=NN.pred, colour='CNN Pred')) +
            geom_errorbar(aes(ymin=NN.lwr, ymax=NN.upr, colour='CNN 95% CI')) + 
            geom_point(aes(y=MCMC.mean, colour='MCMC Estimate')) +
            geom_errorbar(aes(ymin=MCMC.lwr, ymax=MCMC.upr, colour='MCMC 95% HDI')) +
            facet_wrap(~param, scales='free') +
            scale_color_manual(values=c(red, red, blue, blue, black, black))
    )
    
    
    ## no priors, but include MCMC results ##  
  } else if (is.null(priors) && !is.null(MCMC_chains)) {
    
    MCMC_quantiles <- hdi(MCMC_chains)  # 95% HDI of MCMC posteriors
    
    if (log_scale){
      
      MCMC_results <- data.frame(param=unknowns, MCMC.lwr=MCMC_quantiles$CI_low, 
                                 MCMC.upr=MCMC_quantiles$CI_high, 
                                 MCMC.mean=unname(apply(MCMC_chains, 2, mean)))
      
      bootstrap_df <- data.frame(param=unknowns, true=true, NN.pred=pred, 
                                 NN.lwr=lwr, NN.upr=upr) %>%
        merge(MCMC_results, by='param')
      
    } else {  # not log scale
      
      MCMC_results <- data.frame(param=unknowns, MCMC.lwr=exp(MCMC_quantiles$CI_low), 
                                 MCMC.upr=exp(MCMC_quantiles$CI_high), 
                                 MCMC.mean=exp(unname(apply(MCMC_chains, 2, mean))))
      
      bootstrap_df <- data.frame(param=unknowns, true=exp(true), NN.pred=exp(pred), 
                                 NN.lwr=exp(lwr), NN.upr=exp(upr)) %>%
        merge(MCMC_results, by='param')
    }
    
    
    print(ggplot(bootstrap_df %>% group_by(param), aes(x=param)) +
            geom_point(aes(y=true, colour='True'), size=3, shape=18) + 
            geom_point(aes(y=NN.pred, colour='CNN Pred')) +
            geom_errorbar(aes(ymin=NN.lwr, ymax=NN.upr, colour='CNN 95% CI')) + 
            geom_point(aes(y=MCMC.mean, colour='MCMC Estimate')) +
            geom_errorbar(aes(ymin=MCMC.lwr, ymax=MCMC.upr, colour='MCMC 95% HDI')) +
            facet_wrap(~param, scales='free') +
            scale_color_manual(values=c(red, red, blue, blue, black))
    )
    
    
    ## no MCMC results, but include prior bounds in each plot ##
  } else if (!is.null(priors) && is.null(MCMC_chains)) {
    
    if (log_scale){
      
      bootstrap_df <- data.frame(param=unknowns, true=true, NN.pred=pred, 
                                 NN.lwr=lwr, NN.upr=upr) %>%
        merge(priors, by='param')
      
    } else {  # not log scale
      
      bootstrap_df <- data.frame(param=unknowns, true=exp(true), NN.pred=exp(pred), 
                                 NN.lwr=exp(lwr), NN.upr=exp(upr)) %>%
        merge(priors, by='param')
      
      bootstrap_df$prior.lwr = exp(bootstrap_df$prior.lwr)
      bootstrap_df$prior.upr = exp(bootstrap_df$prior.upr)
    }
    
    
    print(ggplot(bootstrap_df %>% group_by(param), aes(x=param)) +
            geom_point(aes(y=true, colour='True'), size=3, shape=18) + 
            geom_errorbar(aes(ymin=prior.lwr, ymax=prior.upr, colour='Prior Bounds')) +
            geom_point(aes(y=NN.pred, colour='CNN Pred')) +
            geom_errorbar(aes(ymin=NN.lwr, ymax=NN.upr, colour='CNN 95% CI')) + 
            facet_wrap(~param, scales='free') +
            scale_color_manual(values=c(red, red, black, black))
    )
    
    
    ## no prior bounds or MCMC results ##
  } else {
    
    if (log_scale){
      bootstrap_df <- data.frame(param=unknowns, true=true, NN.pred=pred, 
                                 NN.lwr=lwr, NN.upr=upr)
    } else {
      bootstrap_df <- data.frame(param=unknowns, true=exp(true), NN.pred=exp(pred), 
                                 NN.lwr=exp(lwr), NN.upr=exp(upr))
    }
    
    print(ggplot(bootstrap_df %>% group_by(param), aes(x=param)) +
            geom_point(aes(y=true, colour='True'), size=3, shape=18) + 
            geom_point(aes(y=NN.pred, colour='CNN Pred')) +
            geom_errorbar(aes(ymin=NN.lwr, ymax=NN.upr, colour='CNN 95% CI')) + 
            facet_wrap(~param, scales='free') +
            scale_color_manual(values=c(red, red, black, black))
    )
  }
}



#' Function for combining parameter test values into a list of dataframes, which can 
#' then be used for plotting in `scatterplot5`.
#'
#' @param test_params list containing three test values for each parameter
#' @param true_params true values of all five parameters

combine_test_cases <- function(test_params, true_params){
  
  delta <- test_params$delta
  P <- test_params$P
  betaEpsilon <- test_params$betaEpsilon
  betaE <- test_params$betaE
  phi <- test_params$phi
  
  delta_true <- true_params$delta
  P_true <- true_params$P
  betaEpsilon_true <- true_params$betaEpsilon
  betaE_true <- true_params$betaE
  phi_true <- true_params$phi
  
  combined_test_cases <- list(
    
    # delta vs P
    data.frame(cbind(delta       = expand.grid(delta, P)$Var1,
                     P           = expand.grid(delta, P)$Var2,
                     betaEpsilon = rep(betaEpsilon_true, 9),
                     betaE       = rep(betaE_true, 9),
                     phi         = rep(phi_true, 9))),
    
    # delta vs betaEpsilon
    data.frame(cbind(delta       = expand.grid(delta, betaEpsilon)$Var1,
                     P           = rep(P_true, 9),
                     betaEpsilon = expand.grid(delta, betaEpsilon)$Var2,
                     betaE       = rep(betaE_true, 9),
                     phi         = rep(phi_true, 9))),
    
    # delta vs betaE
    data.frame(cbind(delta       = expand.grid(delta, betaE)$Var1,
                     P           = rep(P_true, 9),
                     betaEpsilon = rep(betaEpsilon_true, 9),
                     betaE       = expand.grid(delta, betaE)$Var2,
                     phi         = rep(phi_true, 9))),
    
    # delta vs phi
    data.frame(cbind(delta       = expand.grid(delta, phi)$Var1,
                     P           = rep(P_true, 9),
                     betaEpsilon = rep(betaEpsilon_true, 9),
                     betaE       = rep(betaE_true, 9),
                     phi         = expand.grid(delta, phi)$Var2)),
    
    # P vs betaEpsilon
    data.frame(cbind(delta       = rep(delta_true, 9),
                     P           = expand.grid(P, betaEpsilon)$Var1,
                     betaEpsilon = expand.grid(P, betaEpsilon)$Var2,
                     betaE       = rep(betaE_true, 9),
                     phi         = rep(phi_true, 9))),
    
    # P vs betaE
    data.frame(cbind(delta       = rep(delta_true, 9),
                     P           = expand.grid(P, betaE)$Var1,
                     betaEpsilon = rep(betaEpsilon_true, 9),
                     betaE       = expand.grid(P, betaE)$Var2,
                     phi         = rep(phi_true, 9))),
    
    # P vs phi
    data.frame(cbind(delta       = rep(delta_true, 9),
                     P           = expand.grid(P, phi)$Var1,
                     betaEpsilon = rep(betaEpsilon_true, 9),
                     betaE       = rep(betaE_true, 9),
                     phi         = expand.grid(P, phi)$Var2)),
    
    # betaEpsilon vs betaE
    data.frame(cbind(delta       = rep(delta_true, 9),
                     P           = rep(P_true, 9),
                     betaEpsilon = expand.grid(betaEpsilon, betaE)$Var1,
                     betaE       = expand.grid(betaEpsilon, betaE)$Var2,
                     phi         = rep(phi_true, 9))),
    
    # betaEpsilon vs phi
    data.frame(cbind(delta       = rep(delta_true, 9),
                     P           = rep(P_true, 9),
                     betaEpsilon = expand.grid(betaEpsilon, phi)$Var1,
                     betaE       = rep(betaE_true, 9),
                     phi         = expand.grid(betaEpsilon, phi)$Var2)),
    
    # betaE vs phi
    data.frame(cbind(delta       = rep(delta_true, 9),
                     P           = rep(P_true, 9),
                     betaEpsilon = rep(betaEpsilon_true, 9),
                     betaE       = expand.grid(betaE, phi)$Var1,
                     phi         = expand.grid(betaE, phi)$Var2))
  )
  
  combined_test_cases
}



#' Plots predictions from the network for samples generated by different parameter combinations.
#' Each parameter must have three associated test values, then samples are generated using every
#' possible combination of test values. This gives a total of ten 9x9 plots.
#' 
#' @param model the trained neural network
#' @param test_params list containing three test values for each parameter
#' @param true_params true values of all five parameters
#' @param constants values of constants for the blowfly model
#' @param n_samples number of samples to generate for each parameter combination
#' @param log_scale boolean, whether to show all plots in log scale

scatterplot5 <- function(model, test_params, true_params, constants, n_samples, log_scale=FALSE){

  n <- constants$n
  unknowns <- c('delta', 'P', 'betaEpsilon', 'betaE', 'phi')
  
  # obtain list of 10 dataframes, each containing the test cases for a pairwise parameter 
  # combination (e.g. delta vs P)
  param_combinations <- combine_test_cases(test_params, true_params)
  
  for (df_num in 1:length(param_combinations)) {  # looping through each dataframe
    
    df <- param_combinations[[df_num]]
    df$comb <- c(1:9)
    
    # indices of the two parameters in this combination, as listed in `unknowns`
    i <- combn(5, 2)[,df_num][1]
    j <- combn(5, 2)[,df_num][2]
    
    # dataframe to store the predictions
    comb_predictions <- data.frame(array(NA, c(n_samples*9, 3)))
    colnames(comb_predictions) <- c(unknowns[i], unknowns[j], 'comb')
    
    cat('simulating for', 
        c('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth')[df_num], 
        'set of plots... \n')
    
    # looping over test cases
    for (comb in 1:9){
      param_comb <- df[comb,]
      
      # simulate `n_samples` times for this test case
      y_test <- t(  # transpose is needed for compatibility with predict()
        simulate_NBF(delta=param_comb$delta, P=param_comb$P, 
                               betaEpsilon=param_comb$betaEpsilon, betaE=param_comb$betaE, 
                               phi=param_comb$phi, constants=constants, n.reps=n_samples, 
                               standardize=TRUE)
      )
      
      # obtain a prediction from CNN for each sample
      pred <- model %>% predict(y_test, verbose=0)
      
      # convert to regular scale if necessary
      if (!log_scale) pred <- exp(pred)
      
      # store the predictions for this test case
      comb_predictions[(n_samples*(comb-1)+1) : (n_samples*comb), ] = cbind(pred[,i], pred[,j], 
                                                                            rep(comb, n_samples))
    }
    
    if (log_scale){ # plotting in log scale
      
      print(
        ggplot() + geom_point(comb_predictions, mapping=aes(x=!!sym(unknowns[i]),
                                                            y=!!sym(unknowns[j]))) + 
          geom_point(df, mapping=aes(x=log(!!sym(unknowns[i])), y=log(!!sym(unknowns[j])), 
                                     colour='true'), shape=18, size=3) +
          facet_wrap(~comb) +
          xlab(colnames(comb_predictions)[1]) + ylab(colnames(comb_predictions)[2]) +
          scale_x_continuous(sec.axis = sec_axis(~ . , name = paste(colnames(comb_predictions)[2], 'vs',
                                                              colnames(comb_predictions)[1]), 
                                                 breaks = NULL, labels = NULL))
      )
      
    } else {  # plotting in regular scale
      
      print(
        ggplot() + geom_point(comb_predictions, mapping=aes(x=!!sym(unknowns[i]),
                                                            y=!!sym(unknowns[j]))) + 
          geom_point(df, mapping=aes(x=!!sym(unknowns[i]), y=!!sym(unknowns[j]), colour='true'), 
                     shape=18, size=3) +
          facet_wrap(~comb) +
          xlab(colnames(comb_predictions)[1]) + ylab(colnames(comb_predictions)[2]) +
          scale_x_continuous(sec.axis = sec_axis(~ . , name = paste(colnames(comb_predictions)[2], 'vs', 
                                                                    colnames(comb_predictions)[1]), 
                                                 breaks = NULL, labels = NULL))
      )
    }
  }
}  



#' Function to obtain log synthetic likelihood for a sample `y`, given parameter 
#' estimates in `theta`
#' 
#' @param theta current estimate of all five parameters
#' @param constants values of constants for the blowfly model
#' @param y observed blowfly sample for which to calculate the log SL
#' @param n.reps number of replicate data sets to simulate in order to obtain summary statistics
#' @param stats boolean, whether to return the summary statistics instead of the log SL

custom_sl <- function(theta, constants, y, n.reps=500, stats=FALSE){
  
  delta <- theta[1]; P <- theta[2]; betaEpsilon<- theta[3]; betaE <- theta[4]; phi <- theta[5]
  
  # simulate `n.reps` times from model with current parameter values
  Y <- simulate_NBF(delta=delta, P=P, betaEpsilon=betaEpsilon, betaE=betaE, 
                    phi=phi, constants=constants, n.reps=n.reps, standardize=FALSE)
  
  ## now assemble the relevant statistics
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



