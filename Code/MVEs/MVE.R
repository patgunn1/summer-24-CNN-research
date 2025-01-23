library(keras)
library(tidyverse)
source("MVEs/MVE_functions.R")

# simulate training data
sample_size <- 1000
x <- runif(sample_size, 0, 10)
mu <- -x*(x-9)
sigma2 <- ( 0.9 * (x-2) * (x-4) + 1 )^2
y <- rnorm(sample_size, mean=mu, sd=sqrt(sigma2))

# data normalization
gaussian <- TRUE
x_norm <- normalize_data(x, gaussian=gaussian)
y_norm <- normalize_data(y, gaussian=gaussian)

# convert into training arrays
x_train <- array(x_norm, c(sample_size, 1))
y_train <- array(y_norm, c(sample_size, 1))

# y_train <- array(cbind(mu, sigma2), c(sample_size, 2))

# NN parameters
hidden_mean <- c(40, 20, 20, 20, 10, 10, 10, 10)
hidden_var <- c(40, 20, 20, 20, 10, 10, 10, 10)
n_epochs <- 500
batch_size <- 100
lr <- 1e-3
loss <- neg_log_lik
metrics <- NULL
reg_mean <- 0.001
reg_var <- 0.01
verbose1 <- 0
verbose2 <- 0
warmup <- TRUE
fixed_mean <- FALSE

# train the NN
Model <- train_network(x_train, y_train, hidden_mean, hidden_var, n_epochs, 
                       batch_size, lr, loss, metrics, reg_mean, reg_var, 
                       verbose1, verbose2, warmup, fixed_mean)

# predict on test data
x_test <- x_train
pred <- Model %>% predict(x_test)

# denormalize predictions
mu_pred <- denormalize_data(pred[,1], y, gaussian=gaussian)
sigma_pred <- sqrt(exp(pred[,2]) + 1e-7)

if (gaussian){
  sigma_pred <- sigma_pred * sd(y)
} else {
  sigma_pred <- sigma_pred * (max(y) - min(y))
}

# plot results
df <- data.frame(x=x, y=y, mu=mu_pred, sigma=sigma_pred, sigma_true=sqrt(sigma2))
ggplot(df, mapping=aes(x)) + geom_point(aes(y=y)) + 
  geom_line(aes(y=mu, col='mu')) + geom_line(aes(y=sigma, col='sigma')) +
  geom_line(aes(y=sigma_true, col='sigma_true'))


# Regularization notes
regularization_df <- data.frame(x=c('non_constant_mean', 'constant_mean'))
regularization_df$non_constant_sd = list(c('-x*(x-9)', '(0.9*(x-2)*(x-4)+1)^2', 0.001, 0.01), 
                                         c(2, '(0.2*(x-2)*(x-4)+1)^2', 0.1, 0))
regularization_df$constant_sd = list(c('-x*(x-9)', 3, 0, 1), c(5, 1, 0.1, 1))

# Notice how for the below, a sample size of 1000 is not enough
# mu <- 0.4*sin(2*pi*x)
# sigma2 <- ( 0.05 )^2





