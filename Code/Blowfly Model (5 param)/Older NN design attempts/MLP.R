# load packages 
library(invgamma)
library(tidyverse)
library(keras)

source("Blowfly Model (5 param)/functions.R")
set.seed(NULL) # was 66 previously

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

# delta_true <- 0.7
# P_true <- 50
# betaEpsilon_true <- 1
# betaE_true <- 0.1
# phi_true <- 1

# simulate from model and plot
y <- simulate_NBF(delta=delta_true, P=P_true, betaEpsilon=betaEpsilon_true,
                  betaE=betaE_true, phi=phi_true, constants=constants, standardize=FALSE)
df <- data.frame(t=c(1:n), y=y)
ggplot(df) + geom_line(aes(t, y))

n_train <- 2000

# simulate parameter values for training data
delta <- runif(n_train, 0.4*delta_true, 1.6*delta_true)
P <- runif(n_train, 0.4*P_true, 1.6*P_true)
phi <- runif(n_train, 0.4*phi_true, 1.6*phi_true)

theta_train <- array(cbind(delta, P, phi), c(n_train, 3))
log_theta_train <- log(theta_train)

# # simulate using log values?
# delta <- runif(n_train, log(0.4*delta_true), log(1.6*delta_true))
# P <- runif(n_train, log(0.4*P_true), log(1.6*P_true))
# phi <- runif(n_train, log(0.4*phi_true), log(1.6*phi_true))
# log_theta_train <- array(cbind(delta, P, phi), c(n_train, 3))
# theta_train <- exp(log_theta_train)

y_train <- array(NA, c(n_train, n))
for (i in 1:n_train) {
  y_train[i,] <- simulate_NBF(delta=theta_train[i,1], P=theta_train[i,2],
                              betaEpsilon=betaEpsilon_true, betaE=betaE_true,
                              phi=theta_train[i,3], constants=constants, standardize=TRUE)
}

# NN optimization parameters
learning_rate <- 0.01
loss <- 'mse' 
metrics <- NULL
n_epochs <- 250
batch_size <- 100
reg <- 0.01
alpha <- 0.2

# # configure NN
# model <- keras_model_sequential() %>%
#   layer_dense(128, activation = "linear", input_shape = c(n), 
#               kernel_regularizer=regularizer_l2(l=reg)) %>%
#   layer_activation_leaky_relu(alpha=alpha)  %>%
#   layer_dense(64, activation = "relu", kernel_regularizer=regularizer_l2(l=reg)) %>%
#   layer_activation_leaky_relu(alpha=alpha)  %>%
#   layer_dense(16, activation = "linear") %>%
#   layer_activation_leaky_relu(alpha=alpha)  %>%
#   layer_dense(3, activation = "linear")

# configure NN
model <- keras_model_sequential() %>%
  layer_dense(128, activation = "linear", input_shape = c(n)) %>%
  layer_activation_leaky_relu(alpha=alpha)  %>%
  layer_dropout(0.2) %>%
  
  layer_dense(64, activation = "relu") %>%
  layer_activation_leaky_relu(alpha=alpha)  %>%
  layer_dropout(0.2) %>%
  
  layer_dense(16, activation = "linear") %>%
  layer_activation_leaky_relu(alpha=alpha)  %>%
  layer_dropout(0.2) %>%
  
  layer_dense(3, activation = "linear")

summary(model)

# compile model
model %>% compile(
  optimizer_adam(learning_rate = learning_rate), 
  loss = loss, 
  metrics = metrics)

# train model
NN_history <- model %>% fit(
  y_train, log_theta_train,
  epochs = n_epochs,
  batch_size = batch_size,
  validation_split = 0.2,
  verbose = 0)


# testing data
y_test <- array(simulate_NBF(delta=delta_true, P=P_true, betaEpsilon=betaEpsilon_true,
                             betaE=betaE_true, phi=phi_true, constants=constants,
                             standardize=TRUE), c(1, n))
pred <- exp(model %>% predict(y_test))

# bootstrapping technique
bootstrap(model=model, pred=c(pred), param_names=param_names,
          true_params=true_params, n_bootstraps=500, constants=constants)


