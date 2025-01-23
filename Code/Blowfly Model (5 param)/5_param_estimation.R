library(utils)
library(tidyverse)
library(keras3)
library(sl)
library(bayestestR)

start.time.NN <- Sys.time()

source("Blowfly Model (5 param)/CNN_designs.R")
source("Blowfly Model (5 param)/functions.R")

# known parameters
constants <- list(n=300, tau=14, N0=400)

# unknown parameters to be estimated
delta_true <- 0.16
P_true <- 6.5
betaEpsilon_true <- 1
betaE_true <- 0.1
phi_true <- 1

# define boundaries for prior uniform distributions
prior_bounds <- data.frame(param=c('delta', 'P', 'betaEpsilon', 'betaE', 'phi'), prior.lwr=NA, prior.upr=NA)

# rounded versions of the bounds above
prior_bounds[1, 2:3] = c(-3.7, 0)    # delta
prior_bounds[2, 2:3] = c(0.9, 2.9)    # P
prior_bounds[3, 2:3] = c(-1.5, 1.5)   # betaEpsilon
prior_bounds[4, 2:3] = c(-3.5, -1.1)  # betaE
prior_bounds[5, 2:3] = c(-1.5, 1.5)   # phi

# simulate parameter training values, uniformly around log of true values
n_train <- 7000
delta <- runif(n_train, prior_bounds[1,]$prior.lwr, prior_bounds[1,]$prior.upr)
P <- runif(n_train, prior_bounds[2,]$prior.lwr, prior_bounds[2,]$prior.upr)
betaEpsilon <- runif(n_train, prior_bounds[3,]$prior.lwr, prior_bounds[3,]$prior.upr)
betaE <- runif(n_train, prior_bounds[4,]$prior.lwr, prior_bounds[4,]$prior.upr)
phi <- runif(n_train, prior_bounds[5,]$prior.lwr, prior_bounds[5,]$prior.upr)

# combine into array for training
log_theta_train <- array(cbind(delta, P, betaEpsilon, betaE, phi), c(n_train, 5))
theta_train <- exp(log_theta_train)

# simulate training samples
y_train <- array(NA, c(n_train, constants$n))
for (i in 1:n_train) {
  y_train[i,] <- simulate_NBF(delta=theta_train[i,1], P=theta_train[i,2],
                              betaEpsilon=theta_train[i,3], betaE=theta_train[i,4],
                              phi=theta_train[i,5], constants=constants, n.reps=1, 
                              standardize=TRUE)
}


########### fit and train model########### 
CNN <- train_custom_CNN(log_theta_train, y_train, constants, verbose=2,
                        reg=0.0002, dropout_rate=0, n_epochs=75, validation_split=0)

end.time.NN <- Sys.time()

# testing data; keep a record of the unstandardized sample for later M-H MCMC algorithm
y_test_unstandardized <- simulate_NBF(delta=delta_true, P=P_true, betaEpsilon=betaEpsilon_true,
                                      betaE=betaE_true, phi=phi_true, constants=constants, 
                                      n.reps=1, standardize=FALSE)

# then standardize the sample for input into the CNN, and convert into an array
y_test <- array(apply(y_test_unstandardized, 2, standardize.1D), c(1, constants$n))

s1 <- Sys.time()
pred <- CNN %>% predict(y_test) # obtain (log scale) prediction from CNN
s2 <- Sys.time()
s2 - s1

# scatterplots for predictions from different parameter values
test_params <- list(delta       = c(0.10, 0.16, 0.22),
                    P           = c(4, 6.5, 9),
                    betaEpsilon = c(0.6, 1, 1.4),
                    betaE       = c(0.06, 0.1, 0.14),
                    phi         = c(0.6, 1, 1.4))

true_params <- list(delta=delta_true, P=P_true, betaEpsilon=betaEpsilon_true, 
                    betaE=betaE_true, phi=phi_true)

scatterplot5(model=CNN, test_params=test_params, true_params=true_params,
             constants=constants, n_samples=80, log_scale=TRUE)


# bootstrapping technique
unknowns <- c('delta', 'P', 'betaEpsilon', 'betaE', 'phi')

bootstrap(model=CNN, pred=c(pred), unknowns=unknowns, true_params=true_params, 
          n_bootstraps=1000, constants=constants, priors=prior_bounds, log_scale=TRUE)


########### synthetic likelihood ########### 

start.time.MCMC <- Sys.time()

# use the unstandardized sample
y_test <- y_test_unstandardized

# initial MVN check of summary statistics
par(mfrow=c(1,3))
sY <- custom_sl(theta=c(delta_true, P_true, betaEpsilon_true, betaE_true, phi_true), 
                constants=constants, y=y_test, n.rep=500, stats=TRUE)
sy <- attr(sY,"observed")
MVN.check(sY,sy)


#### number of Monte Carlo iterations ####
n.mc <- 50000

th <- matrix(0,5,n.mc)  # storage for parameters
rss <- llr <- rep(NA,n.mc)  # storage for l_s and diagnostic

theta0 <- c(0.1, 5, 1.3, 0.06, 0.6)  # ok start (long way from truth)

th[,1] <- log(theta0)  # transform to log scale
prop.sd <- c(0.07, 0.1, 0.07, 0.07, 0.1)  # proposal standard deviation

pind <- c(1:5)
thetap <- exp(th)  # params on original scale

# initial l_s
# llt <- attr(custom_sl(theta=thetap, constants=constants, y=y_test, n.rep=500, stats=FALSE), 'true')
llt <- custom_sl(theta=thetap, constants=constants, y=y_test, n.rep=500, stats=FALSE)
llr[1] <- attr(llt,"true")
reject <- 0
uni <- runif(n.mc)
ll <- rep(-100, 5)
ul <- rep(100, 5)

for (i in 2:n.mc) {  # main MCMC loop
  # make proposal
  th[pind,i] <- th[pind,i-1] + rnorm(5)*prop.sd
  thetap <- exp(th[,i])
  
  if (i%%1000==0) cat(' On proposal number', i, 'current estimate is:', thetap, '\n',
                      'Time elapsed: ', Sys.time() - start.time.MCMC, '\n')
  
  # l_s for proposal...
  llp <- custom_sl(theta=thetap, constants=constants, y=y_test, n.rep=500, stats=FALSE)
  # llp <- attr(custom_sl(theta=thetap, constants=constants, y=y_test, n.rep=500, stats=FALSE), 'true')
  
  # get acceptance probability
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
}

end.time.MCMC <- Sys.time()

# store chain evolutions in a dataframe
param_chains <- data.frame(t(th))
names(param_chains) <- unknowns

# plot evolution of chains for each parameter, plus evolution of l_s
ggplot(gather(param_chains) %>% mutate(t = rep(c(1:n.mc), 5)) %>% 
         rbind(data.frame(key=rep('l_s', n.mc), value=llr, t=c(1:n.mc))), 
       aes(x=t, y=value)) + geom_line() + 
  geom_hline(aes(yintercept=value), gather(log(data.frame(true_params))), linetype='dashed', 
             colour='#F8766D') +
  facet_wrap(~key, scales = 'free_y')

# plot histograms showing approx posteriors for each parameter
ggplot(gather(param_chains), aes(value)) +
  geom_histogram(bins=75) + 
  geom_vline(aes(xintercept=value), gather(log(data.frame(true_params))), linetype='dashed', 
             colour='#F8766D') +
  facet_wrap(~key, scales = 'free_x')

# plot priors, NN and MCMC intervals all together
bootstrap(model=CNN, pred=c(pred), unknowns=unknowns, true_params=true_params, n_bootstraps=1000, 
          constants=constants, priors=prior_bounds, MCMC_chains=param_chains, log_scale=TRUE)

cat(' Neural Network time: ', end.time.NN - start.time.NN, '\n',
    'MCMC algorithm time: ', end.time.MCMC - start.time.MCMC, '\n',
    'Acceptance rate: ', (1-reject/n.mc) * 100, '% \n',
    'Final estimate: ', exp(th[,n.mc]), '\n',
    'Means of each chain: ', apply(exp(th), 1, mean), '\n')

# save(th, file='param_chains.Rdata')


