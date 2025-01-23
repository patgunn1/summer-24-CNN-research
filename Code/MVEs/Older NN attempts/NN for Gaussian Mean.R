library(keras)
library(tidyverse)

# Set variables
sigma <- 2
n_train <- 1000
sample_size <- 5000

# Simulate train data
theta_train <- array(runif(n_train, min=0.1, max=10), c(n_train, 1))
y_train <- array(NA, c(n_train, sample_size, 1))

for (i in 1:n_train) {
  y <- rnorm(sample_size, mean=theta_train[i,], sd=sigma)
  y_train[i,,] = y
}

# Configure NN
model <- keras_model_sequential() %>%
  layer_dense(64, activation = "relu", input_shape = c(sample_size), 
              name = "hidden1") %>%
  layer_dense(16, activation = "relu", name = "hidden2") %>%
  layer_dense(8, activation = "relu", name = "hidden3") %>%
  layer_dense(1, name = "output", activation = "linear")

# Set optimization parameters
learning_rate = 0.0001
loss          = 'mse'   # mean squared error regression
metrics       = 'mae'   # mean absolute error
n_epochs      = 30
batch_size    = 32

summary(model)

# Compile model
model %>% compile(
  optimizer_adam(learning_rate = learning_rate), 
  loss = loss, 
  metrics = metrics)

# Train model
NN_history <- model %>% fit(
  y_train, theta_train,
  epochs = n_epochs,
  batch_size = batch_size,
  validation_split = 0.2,
  verbose = 0)

# Simulate testing data and predict
n_test <- 25
thetas <- runif(n_test, min=3, max=7)

theta_test <- array(thetas, c(n_test, 1))
y_test <- array(NA, c(n_test, sample_size, 1))
for (i in 1:n_test) {
  y <- rnorm(sample_size, mean=theta_test[i,], sd=sigma)
  y_test[i,,] = y
}

theta_pred <- model %>% predict(y_test)


# Bootstrapping samples

n_bootstraps <- 500
y_bootstrap <- array(NA, c(n_test, sample_size, n_bootstraps))
for (j in 1:n_bootstraps) {
  for (i in 1:n_test) {
    y <- rnorm(sample_size, mean=theta_pred[i,], sd=sigma)
    y_bootstrap[i,,j] = y
  }
}

theta_bootstrap <- array(NA, c(n_test, n_bootstraps))
for (j in 1:n_bootstraps) {
  theta <- model %>% predict(y_bootstrap[,,j])
  theta_bootstrap[,j] <- theta
}

upr <- numeric(n_test)
lwr <- numeric(n_test)
for (i in 1:n_test) {
  interval <- unname(quantile(theta_bootstrap[i,], probs = c(0.025, 0.975)))
  lwr[i] <- interval[1]
  upr[i] <- interval[2]
}


# Calculating MLE and plotting confidence intervals
mle <- numeric(n_test)
for (i in 1:n_test){
  mle[i] = mean(y_test[i,,])
}
z <- qnorm(0.975)

# bootstrap_df <- data.frame(sample_num = c(1:n_test), true = thetas, lwr = lwr, upr = upr)
# ggplot(bootstrap_df, aes(x=sample_num, y=true, ymin=lwr , ymax=upr)) +
#   geom_point(size = 1.1) + geom_errorbar(width = 0.3, color='orange') +
#   ylab('Mean') + xlab('Sample')

bootstrap_df <- data.frame(sample_num = c(1:n_test), true = thetas, 
                           lwr_mle = mle-z*sigma/sqrt(sample_size), 
                           upr_mle = mle+z*sigma/sqrt(sample_size),
                           lwr_nn = lwr, upr_nn = upr)

ggplot(bootstrap_df, aes(x=sample_num)) + geom_point(aes(y=true, colour='true mean'), size = 1) +
  geom_errorbar(aes(ymin=lwr_nn, ymax=upr_nn, color='bootstrap interval'), width = 0.3) +
  geom_errorbar(aes(ymin=lwr_mle, ymax=upr_mle, color='MLE interval'), width = 0.3) +
  scale_color_manual(values=c('true mean'='black', 'bootstrap interval'='orange', 'MLE interval'='blue')) + 
  ylab('Mean') + xlab('Sample')



