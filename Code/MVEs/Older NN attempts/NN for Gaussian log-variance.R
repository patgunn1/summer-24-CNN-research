library(keras)

# Set variables
mu <- 5
n_train <- 2000
sample_size <- 1000

# Simulate train data
theta_train <- array(runif(n_train, min=1e-7, max=log(5)), c(n_train, 1))
y_train <- array(NA, c(n_train, sample_size, 1))

for (i in 1:n_train) {
  y <- rnorm(sample_size, mean=mu, sd=sqrt(exp(theta_train[i,])))
  y <- y - mu  # standardizing the data?
  y_train[i,,] = y
}

# Configure NN
# model <- keras_model_sequential() %>%
#   layer_dense(32, activation = "relu", input_shape = c(sample_size), 
#               kernel_regularizer = regularizer_l2(l = 0.01), 
#               name = "hidden1") %>%
#   layer_dropout(0.1) %>%
#   layer_dense(16, activation = "relu", input_shape = c(sample_size), 
#               kernel_regularizer = regularizer_l2(l = 0.01), 
#               name = "hidden2") %>%
#   layer_dropout(0.1) %>%
#   layer_dense(1, name = "output", activation = "linear")

model <- keras_model_sequential() %>%
  layer_dense(8, activation = "relu", input_shape = c(sample_size), 
              name = "hidden1") %>%
  layer_dropout(0.4) %>%
  layer_dense(4, activation = "relu", kernel_regularizer = regularizer_l2(l = 0.1), 
              name = "hidden2") %>%
  layer_dropout(0.4) %>%
  layer_dense(4, activation = "relu", name = "hidden3") %>%
  layer_dropout(0.4) %>%
  layer_dense(1, name = "output", activation = "linear")

# Set optimization parameters
learning_rate = 0.01
loss          = 'mse'   # mean squared error regression
metrics       = 'mae'   # mean absolute error
n_epochs      = 40
batch_size    = 100

# Compile model
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = loss, 
  metrics = metrics)

# Train model
NN_history <- model %>% fit(
  y_train, theta_train,
  epochs = n_epochs,
  batch_size = batch_size,
  validation_split = 0.2,
  verbose = 2)

# Simulate testing data and predict
n_test <- 5
thetas <- runif(n_test, 1.5, 3.5)

theta_test <- array(log(thetas), c(n_test, 1))
y_test <- array(NA, c(n_test, sample_size, 1))

for (i in 1:n_test) {
  y <- rnorm(sample_size, mean=mu, sd=sqrt(thetas[i]))
  y <- y - mu  # standardizing the data?
  y_test[i,,] = y
}
  
theta_pred <- model %>% predict(y_test)
print(exp(theta_pred))
print(thetas)

print(exp(theta_train[1:3, 1]))
print(exp(model %>% predict(y_train[1:3,,])))


## Bootstrapping samples ##

# n_bootstraps <- 500
# y_bootstrap <- array(NA, c(n_test, sample_size, n_bootstraps))
# for (j in 1:n_bootstraps) {
#   for (i in 1:n_test) {
#     y <- rnorm(sample_size, mean=mu, sd=sqrt(exp(theta_pred[i,])))
#     y_bootstrap[i,,j] = y
#   }
# }
# 
# theta_bootstrap <- array(NA, c(n_test, n_bootstraps))
# for (j in 1:n_bootstraps) {
#   theta <- model %>% predict(y_bootstrap[,,j])
#   theta_bootstrap[,j] <- theta
# }
# 
# upr <- numeric(n_test)
# lwr <- numeric(n_test)
# for (i in 1:n_test) {
#   interval <- unname(quantile(theta_bootstrap[i,], probs = c(0.025, 0.975)))
#   lwr[i] <- interval[1]
#   upr[i] <- interval[2]
# }
# 
# # Plotting confidence intervals
# 
# bootstrap_df <- data.frame(sample_num = c(1:n_test), true = thetas, lwr = lwr, upr = upr)
# 
# ggplot(bootstrap_df, aes(x=sample_num, y=true, ymin=lwr , ymax=upr)) +
#   geom_point(size = 1.1) + geom_errorbar(width = 0.3, color='orange') +
#   ylab('Mean') + xlab('Sample')



