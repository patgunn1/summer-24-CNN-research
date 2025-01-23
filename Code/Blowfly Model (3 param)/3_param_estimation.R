library(utils)
library(tidyverse)
library(keras3)
library(sl)

start.time.NN <- Sys.time()

source("Blowfly Model (3 param)/CNN_designs.R")
source("Blowfly Model (3 param)/functions.R")

# known parameters
constants <- list(n=300, tau=14, N0=400)

betaEpsilon_true <- 1
betaE_true <- 0.1

# 'unknown' parameters to be estimated
delta_true <- 0.16
P_true <- 6.5
phi_true <- 1

# define boundaries for prior uniform distributions
prior_bounds <- data.frame(param=c('delta', 'P', 'phi'), prior_lwr=NA, prior_upr=NA)

prior_bounds[1, 2:3] = c( 2*log(delta_true), 0 )               # delta bounds
prior_bounds[2, 2:3] = c( 0.5*log(P_true), 1.5*log(P_true) )   # P bounds
prior_bounds[3, 2:3] = c(-1.5, 1.5)                            # phi bounds

# simulate parameter training values, uniformly around log of true values
n_train <- 5000
delta <- runif(n_train, prior_bounds[1,]$prior_lwr, prior_bounds[1,]$prior_upr)
P <- runif(n_train, prior_bounds[2,]$prior_lwr, prior_bounds[2,]$prior_upr)
phi <- runif(n_train, prior_bounds[3,]$prior_lwr, prior_bounds[3,]$prior_upr)

log_theta_train <- array(cbind(delta, P, phi), c(n_train, 3))
theta_train <- exp(log_theta_train)

# simulate training samples
y_train <- array(NA, c(n_train, constants$n))
for (i in 1:n_train) {
  y_train[i,] <- simulate_NBF(delta=theta_train[i,1], P=theta_train[i,2],
                              betaEpsilon=betaEpsilon_true, betaE=betaE_true,
                              phi=theta_train[i,3], constants=constants, 
                              n.reps=1, standardize=TRUE)
}


########### fit and train model########### 
CNN <- train_custom_CNN(log_theta_train, y_train, constants, verbose=2, 
                        reg=0.0001, n_epochs=40)


# testing data; keep a record of the unstandardized sample for later M-H MCMC algorithm
y_test_unstandardized <- array(simulate_NBF(delta=delta_true, P=P_true, 
                                            betaEpsilon=betaEpsilon_true, betaE=betaE_true, 
                                            phi=phi_true, constants=constants, n.reps=1, 
                                            standardize=FALSE), c(1, constants$n))

# then standardize the sample for input into the CNN
y_test <- (y_test_unstandardized - min(y_test_unstandardized)) / 
          (max(y_test_unstandardized) - min(y_test_unstandardized))


pred <- exp(CNN %>% predict(y_test))  # obtain prediction from CNN


# bootstrapping technique
true_params <- list(delta=delta_true, P=P_true, betaEpsilon=betaEpsilon_true, 
                    betaE=betaE_true, phi=phi_true)
unknowns <- c('delta', 'P', 'phi')

bootstrap(model=CNN, pred=c(pred), unknowns=unknowns, true_params=true_params, 
          n_bootstraps=500, constants=constants, priors=prior_bounds, log_scale=TRUE)


# scatterplots for predictions from different parameter values
delta_test <- c(0.10, 0.16, 0.22)
P_test <- c(4, 6.5, 9)
phi_test <- c(0.6, 1, 1.4)

scatterplot3(model=CNN, delta=delta_test, P=P_test, phi=phi_test, true_params=true_params, 
             constants=constants, n_samples=70, log_scale=TRUE)

end.time.NN <- Sys.time()

########### synthetic likelihood ########### 

start.time.MCMC <- Sys.time()

# use the unstandardized sample
y_test <- y_test_unstandardized

if (!is.null(dim(y_test))) y_test <- y_test[1,]  # convert to a vector if needed
true_betas <- c(betaEpsilon_true, betaE_true)

# initial MVN check of summary statistics
par(mfrow=c(2,2))
sY <- custom_sl(theta=c(delta_true, P_true, phi_true), constants=constants, 
                true_betas=true_betas, y=y_test, n.rep=500, stats=TRUE)
sy <- attr(sY,"observed")
MVN.check(sY,sy)  

n.mc <- 50000  # number of Monte Carlo iterations 

th <- matrix(0,3,n.mc)  # storage for parameters
rss <- llr <- rep(NA,n.mc)  # storage for l_s and diagnostic

theta0 <- c(0.1, 4, 0.6)  # ok start (long way from truth)

th[,1] <- log(theta0)  # transform to log scale
prop.sd <- c(0.025, 0.06, 0.06)  #proposal standard deviation
pind <- c(1:3)
thetap <- exp(th)  # params on original scale

# initial l_s
llt <- custom_sl(theta=thetap, constants=constants, 
                 true_betas=true_betas, y=y_test, n.rep=500, stats=FALSE)
llr[1] <- attr(llt,"true")
reject <- 0
uni <- runif(n.mc)
ll <- c(-100, -100,-100)
ul <- c(100,100,100)

for (i in 2:n.mc) {  # main MCMC loop
  # make proposal....
  th[pind,i] <- th[pind,i-1] + rnorm(3)*prop.sd
  thetap <- exp(th[,i])
  
  if (i%%5000==0) cat('On proposal number', i, 'current estimate is', thetap, '\n')
  
  # l_s for proposal...
  llp <- custom_sl(theta=thetap, constants=constants, 
            true_betas=true_betas, y=y_test, n.rep=100, stats=FALSE)
  
  # get acceptance probability...
  
  alpha <- min(1,exp(llp-llt))
  if (sum(th[,i]<ll|th[,i]>ul)) alpha <- 0
  
  # accept/reject...
  if (uni[i]>alpha) { # reject
    th[,i] <- th[,i-1]
    llr[i] <- llr[i-1]
    reject <- reject + 1
  } else { # accept
    llr[i] <- attr(llp,"true")
    llt <- llp
  }
  ## Plot results every so often....
  # if (i%%200==0) {
  #   ind <- 1:i
  #   if (i>1000) ind2 <- round(i/2):i else ind2 <- ind ## for plot range 
  #   par(mfrow=c(3,3),mar=c(4,4,1,1))
  #   for (k in 1:6) plot(ind,th[k,ind],type="l",ylim=range(th[k,ind2]),ylab=pname[k])
  #   plot(ind,llr[ind],type="l",ylim=range(llr[ind2]))
  #   ##plot(ind,rss[ind],type="l",ylim=range(rss[ind2]),main=round(1-reject/i,digits=2))
  #   thetap <- c(exp(th[1:3,i]),th[4,i],exp(th[5:6,i]))
  #   Y <- des.bf(thetap,burn.in,n.y*step,n.reps)[1:n.y*step,]
  #   lim <- range(bf$pop)
  #   plot(1:n.y,Y[,1],type="l",ylim=lim)
  #   lines(1:n.y,bf$pop,col=2)
  #   lines(1:n.y,Y[,2],type="l",ylim=lim,lty=2)
  # }
}

thetap <- exp(th[,n.mc])
chain_means <- c(mean(exp(th[1,])), mean(exp(th[2,])), mean(exp(th[3,])))
param_chains <- data.frame(delta=exp(th[1,]), P=exp(th[2,]), phi=exp(th[3,]))

ggplot(gather(param_chains), aes(value)) +
  geom_histogram(bins=75) + facet_wrap(~key, scales = 'free_x')

ggplot(gather(param_chains) %>% mutate(t = rep(c(1:n.mc), 3)), 
       aes(x=t, y=value)) +
  geom_line() + facet_wrap(~key, scales = 'free_y')

end.time.MCMC <- Sys.time()

cat(' Neural Network time: ', end.time.NN - start.time.NN, '\n',
    'MCMC algorithm time: ', end.time.MCMC - start.time.MCMC, '\n',
    'Acceptance rate: ', (1-reject/n.mc) * 100, '% \n',
    'Final estimate: ', thetap, '\n',
    'Means of each chain: ', chain_means, '\n')





