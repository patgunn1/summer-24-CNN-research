library(keras)

## First approach ##

# mu <- 5
# n_train <- 100
# theta_train <- runif(n_train, min=0.1, max=2)
# y_train <- rnorm(n_train, mean=mu, sd=theta_train)
# y_train <- (y_train - mu)/(sd(y_train))  # standardizing the data?


## New approach ##
# Set variables
mu <- 5
n_train <- 10000
sample_size <- 1000

# Simulate train data
theta_train <- array(runif(n_train, min=0.01, max=3), c(n_train, 1))
y_train <- array(NA, c(n_train, sample_size, 1))

for (i in 1:n_train) {
  y <- rnorm(sample_size, mean=mu, sd=theta_train[i,])
  y <- y - mu  # standardizing the data?
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
learning_rate = 0.001
loss          = 'mse'   # mean squared error regression
metrics       = 'mae'   # mean absolute error
n_epochs      = 30
batch_size    = 64

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
  verbose = 2)


# Simulate testing data and predict
thetas <- c(0.8, 1, 1.5, 1.9)
n_test <- length(thetas)

theta_test <- array(thetas, c(n_test, 1))
y_test <- array(NA, c(n_test, sample_size, 1))

for (i in 1:n_test) {
  y <- rnorm(sample_size, mean=mu, sd=theta_test[i,])
  y <- (y - mu)  # standardizing the data?
  y_test[i,,] = y
}
  
theta_pred <- model %>% predict(y_test)
print(theta_pred)


