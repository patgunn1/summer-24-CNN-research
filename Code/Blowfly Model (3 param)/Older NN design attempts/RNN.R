# load packages 
library(invgamma)
library(tidyverse)
library(keras)

source("Nicholson's Blowfly Model/functions.R")

# known parameters
n <- 300
tau <- 14
N0 <- 400
constants <- list(n=n, tau=tau, N0=N0)

# 'unknown' parameters
delta_true <- 0.16
P_true <- 6.5
betaEpsilon_true <- 1
betaE_true <- 0.1
phi_true <- 1

# define boundaries for prior uniform distributions
prior_bounds <- data.frame(param=c('delta', 'P', 'phi'), prior_lwr=NA, prior_upr=NA)

prior_bounds[1, 2:3] = c( 2*log(delta_true), 0 )               # delta bounds
prior_bounds[2, 2:3] = c( 0.5*log(P_true), 1.5*log(P_true) )   # P bounds
prior_bounds[3, 2:3] = c(-1.5, 1.5)                            # phi bounds

# simulate uniformly around log values
n_train <- 2000

delta <- runif(n_train, prior_bounds[1,]$prior_lwr, prior_bounds[1,]$prior_upr)
P <- runif(n_train, prior_bounds[2,]$prior_lwr, prior_bounds[2,]$prior_upr)
phi <- runif(n_train, prior_bounds[3,]$prior_lwr, prior_bounds[3,]$prior_upr)

log_theta_train <- array(cbind(delta, P, phi), c(n_train, 3))
theta_train <- exp(log_theta_train)

y_train <- array(NA, c(n_train, n))
for (i in 1:n_train) {
  y_train[i,] <- simulate_NBF(delta=theta_train[i,1], P=theta_train[i,2],
                              betaEpsilon=betaEpsilon_true, betaE=betaE_true,
                              phi=theta_train[i,3], constants=constants, standardize=TRUE)
}

# NN optimization parameters
learning_rate <- 0.001
loss <- 'mse' 
metrics <- NULL
n_epochs <- 40
batch_size <- 100
l2 <- regularizer_l2
reg <- 0.01
alpha <- 0.1

# # Recurrent NN
# model <- keras_model_sequential() %>%
#   layer_simple_rnn(units=32, input_shape=c(n,1), activation='tanh', return_sequences=FALSE) %>%
#   layer_dense(units = 8, activation="linear", kernel_regularizer=regularizer_l2(l=reg)) %>%
#   layer_activation_leaky_relu(alpha=alpha) %>%
#   layer_dense(units = 3, activation = "linear")

# LSTM network
# model <- keras_model_sequential() %>%
#   layer_lstm(64, input_shape=c(n,1), activation='tanh') %>%
#   layer_dropout(0.2) %>%
# 
#   layer_dense(units = 8, activation="linear") %>%
#   layer_activation_leaky_relu(alpha=alpha) %>%
#   
#   layer_dense(units = 3, activation = "linear")

# LSTM network 2
model <- keras_model_sequential() %>%
  layer_lstm(8, input_shape=c(n,1), return_sequences=FALSE) %>%

  layer_dense(units = 8, activation="linear") %>%
  layer_activation_leaky_relu(alpha=alpha)  %>%

  layer_dense(units = 3, activation = "linear")

summary(model)

# compile model
model %>% compile(
  optimizer_adam(learning_rate=learning_rate), 
  loss=loss, 
  metrics=metrics)

# train model
NN_history <- model %>% fit(
  y_train, log_theta_train,
  epochs = n_epochs,
  batch_size = batch_size,
  validation_split = 0.2,
  verbose = 2)


# testing data
y_test <- array(simulate_NBF(delta=delta_true, P=P_true, betaEpsilon=betaEpsilon_true,
                             betaE=betaE_true, phi=phi_true, constants=constants,
                             standardize=TRUE), c(1, n))
pred <- exp(model %>% predict(y_test))

# bootstrapping technique
true_params <- c(delta_true, P_true, betaEpsilon_true, betaE_true, phi_true)
unknowns <- c('delta', 'P', 'phi')

bootstrap(model=model, pred=c(pred), unknowns=unknowns, true_params=true_params, 
          n_bootstraps=500, constants=constants, priors=prior_bounds, log_scale=FALSE)






