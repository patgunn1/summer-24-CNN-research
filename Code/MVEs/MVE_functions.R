#### function definitions ####

# negated log-likelihood
neg_log_lik <- function(targets, outputs){
  mu <- outputs[,1]
  var <- exp(outputs[,2]) + 1e-7
  y <- targets[,1]
  
  log(var) + (y-mu)^2 / var
}


# train an MVE
train_network <- function(x_train, y_train, hidden_mean, hidden_var, n_epochs, 
                          batch_size, lr=0.01, loss=neg_lok_lik, metrics=NULL, 
                          reg_mean=0, reg_var=0, verbose1=0, verbose2=0, 
                          warmup=TRUE, fixed_mean=FALSE){
  
  inputs <- layer_input(shape=c(1))
  
  inter_mean <- layer_dense(units=hidden_mean[1], activation="elu",
                            kernel_regularizer=regularizer_l2(l = reg_mean),
                            name="mean")(inputs)
  for (i in 2:length(hidden_mean)) {
    inter_mean <- layer_dense(units=hidden_mean[i], activation="elu", 
                              kernel_regularizer=regularizer_l2(l = reg_mean), 
                              name=sprintf("mean%s", i))(inter_mean)
  }
  
  inter_var <- layer_dense(units=hidden_var[1], activation="elu",
                           kernel_regularizer=regularizer_l2(l = reg_var),
                           name="var")(inputs)
  for (i in 2:length(hidden_var)) {
    inter_var <- layer_dense(units=hidden_var[i], activation="elu", 
                             kernel_regularizer=regularizer_l2(l = reg_var), 
                             name=sprintf("var%s", i))(inter_var)
  }
  
  output_mean <- layer_dense(units=1, activation="linear",
                             kernel_regularizer=regularizer_l2(l = reg_mean),
                             name="meanout")(inter_mean)
  output_var <- layer_dense(units=1, activation="linear",
                            kernel_regularizer=regularizer_l2(l = reg_var),
                            bias_initializer = initializer_constant(1.0),
                            name="varout")(inter_var)
  
  outputs <- layer_concatenate(c(output_mean, output_var))
  model <- keras_model(inputs, outputs)
  
  if (!warmup){
    model %>% compile(
      optimizer = optimizer_adam(learning_rate=lr, clipvalue=5), 
      loss = loss, 
      metrics = metrics)
    
    model %>% fit(
      x_train, y_train,
      epochs = n_epochs,
      batch_size = batch_size,
      validation_split = 0.2,
      verbose = verbose2)
    
  } else {
    
    # freeze variance layers
    for (layer in model$layers) {
      if (substring(layer$name, 1, 1) == 'v'){
        layer$trainable = FALSE
      }
    }
    
    model %>% compile(
      optimizer = optimizer_adam(learning_rate=lr, clipvalue=5), 
      loss = loss, 
      metrics = metrics)
    
    model %>% fit(
      x_train, y_train,
      epochs = n_epochs,
      batch_size = batch_size,
      validation_split = 0.2,
      verbose = verbose1)
    
    pred <- model %>% predict(x_train)
    logmse <- log(mean((pred[,1] - y_train)^2))
    
    varout <- model$layers[[length(model$layers) - 1]]
    set_weights(varout, list(get_weights(varout)[[1]], array(logmse)))
    
    for (layer in model$layers) {
      layer$trainable = TRUE
    }
    
    if (fixed_mean){
      for (layer in model$layers) {
        if (substring(layer$name, 1, 1) == 'm'){
          layer$trainable = FALSE
        }
      }
    }
    
    model %>% compile(
      optimizer = optimizer_adam(learning_rate=lr, clipvalue=5), 
      loss = loss, 
      metrics = metrics)
    
    model %>% fit(
      x_train, y_train,
      epochs = n_epochs,
      batch_size = batch_size,
      validation_split = 0.2,
      verbose = verbose2)
    
  }
  model
}


# data normalization
normalize_data <- function(x, gaussian=TRUE){
  if (gaussian){
    mu <- mean(x)
    sigma <- sd(x)
    normalized_x <- (x - mu)/sigma
    } 
  else {
    x_min <- min(x)
    x_max <- max(x)
    normalized_x <- (x - x_min) / (x_max - x_min)
    }
  normalized_x
} 


# data denormalization
denormalize_data <- function(x, y, gaussian=TRUE){
  if (gaussian){
    denormalized_x <- x * sd(y) + mean(y)
    } 
  else {
    ymin <- min(y)
    ymax <- max(y)
    denormalized_x <- x * (ymax - ymin) + ymin
  }
  denormalized_x
}











