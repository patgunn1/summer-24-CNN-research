#' Function for designing, compiling and training Amanda's CNN
#' 
#' @param log_theta_train log values of theta training samples
#' @param y_train standardized y training samples
#' @param constants list containing values of the constants for the blowfly model
#' @param verbose verbosity when training the CNN
#' @param reg value of regularisation constant in the second 1D convolution layer
#'
#' All other params are for the design, compiling and training of the CNN, and can be left as
#' their default values

train_amandas_CNN <- function(log_theta_train, y_train, constants, verbose=2, reg=0.01,
                              learning_rate=0.01,
                              loss='mse',
                              metrics=NULL,
                              n_epochs=75,
                              batch_size=128,
                              kernel_size=c(5),
                              pool_size=c(2),
                              relu_leak=0.1,
                              padding='same',
                              validation_split=0.2){
  
  n <- constants$n
  
  # model architecture
  model <- keras_model_sequential() %>%
    layer_conv_1d(filters = 128, kernel_size = kernel_size, padding = padding, input_shape = c(n,1),
                  activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_max_pooling_1d(pool_size, padding = padding)  %>%

    layer_conv_1d(filters = 128, kernel_size = kernel_size, padding = padding,
                  activation = "linear", kernel_regularizer=regularizer_l2(reg)) %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_max_pooling_1d(pool_size, padding = padding)  %>%

    layer_conv_1d(filters = 16, kernel_size = kernel_size, padding = padding,
                  activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_max_pooling_1d(pool_size, padding = padding)  %>%
    layer_flatten() %>%

    layer_dense(units = 4, activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak)  %>%
    layer_dense(units = 8, activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_dense(units = 16, activation = "linear")  %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_dense(units = 3, activation = "linear")
  
  
  summary(model)
  
  # compile model
  model %>% compile(
    optimizer_adam(learning_rate = learning_rate), 
    loss = loss, 
    metrics = metrics)
  
  # train model
  model %>% fit(
    y_train, log_theta_train,
    epochs = n_epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    verbose = verbose)
  
  return(model)
}



#' Function for designing, compiling and training my custom CNN, inspired by Amanda's
#'
#' @param log_theta_train log values of theta training samples
#' @param y_train standardized y training samples
#' @param constants list containing values of the constants for the blowfly model
#' @param verbose verbosity when training the CNN
#' @param reg value of regularisation constant in the first dense layer
#'
#' All other params are for the design, compiling and training of the CNN, and can be left as
#' their default values

train_custom_CNN <- function(log_theta_train, y_train, constants, verbose=2, reg=0.001,
                             learning_rate=0.01,
                             loss='mse',
                             metrics=NULL,
                             n_epochs=75,
                             batch_size=128,
                             kernel_size=c(5),
                             pool_size=c(2),
                             relu_leak=0.1,
                             padding='same',
                             dropout_rate=0.2,
                             validation_split=0.2){
  
  n <- constants$n
  
  # model architecture
  model <- keras_model_sequential() %>%
    layer_conv_1d(filters = 64, kernel_size = kernel_size, padding = padding, input_shape = c(n,1),
                  activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_max_pooling_1d(pool_size, padding = padding)  %>%
    
    layer_conv_1d(filters = 64, kernel_size = kernel_size, padding = padding,
                  activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_dropout(dropout_rate) %>%
    layer_max_pooling_1d(pool_size, padding = padding)  %>%
    layer_flatten() %>%
    
    layer_dense(units = 64, activation = "linear", kernel_regularizer = regularizer_l2(reg)) %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_dropout(dropout_rate) %>%
    
    layer_dense(units = 32, activation = "linear")  %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    
    layer_dense(units = 3, activation = "linear")
  
  
  summary(model)
  
  # compile model
  model %>% compile(
    optimizer_adam(learning_rate = learning_rate), 
    loss = loss, 
    metrics = metrics)
  
  # train model
  model %>% fit(
    y_train, log_theta_train,
    epochs = n_epochs,
    batch_size = batch_size,
    validation_split = validation_split,
    verbose = verbose)
  
  return(model)
}




