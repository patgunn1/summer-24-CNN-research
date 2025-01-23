

## ----------------
## Simulate Brown-Resnick processes
## ----------------

simu_samp = function(data_y, n.rep.test, nn, coord, cov.type){
  
  simu.data =  array(NA, c(dim = nrow(data_y), n.rep.test, nn, nn))
  
  for(ii in 1:nrow(data_y)){
    for(irep in 1:n.rep.test){
      simu.data[ii,irep,,] <- matrix(rmaxstab(1, coord, cov.mod = cov.type,
                                           range = data_y[ii, 1], 
                                           smooth =  data_y[ii, 2]), nn, nn)
    }
    print(ii)
  }
  
  return(simu.data = simu.data)
}

## ----------------
## Fit a Brown-Resnick process to data
## ----------------

fit_BR = function(coords, data, dist_weig, n.rep.test, s.init, r.init){
  
  fit.params = matrix(NA, n.rep.test, 2)
  fit.like = rep(NA, n.rep.test)
  for(irep in 1:n.rep.test){
    fit = fitmaxstab(t(data[,irep]), coords, "brown",
                     weights = dist_weig, 
                     method = "L-BFGS-B", 
                     #control = list(pgtol = 1e-20),
                     start = list(range = r.init,smooth = s.init))
    
    fit.like[irep] = fit$opt.value
    fit.params[irep,] = fit$param[1:2] 
  }
  
  out = list(fit.like = fit.like, fit.params = fit.params)
  return(out)
}


## ----------------
## Fit the CNN and predict test 
## ----------------

fit_pred_CNN = function(data_x_train, data_y_train, data_x_pred){
  
  nn = dim(data_x_pred)[2]
  
  x_train <- array_reshape(log(data_x_train), c(nrow(data_x_train), nn, nn, 1))
  y_train = cbind(log(data_y_train[,1]), log(data_y_train[,2]/(2 - data_y_train[,2])))
  x_test <- array_reshape(log(data_x_pred), c(nrow(data_x_pred), nn, nn, 1))
  
  # Define a few parameters to be used in the CNN model
  kernel_size = c(3,3)
  pool_size   = c(2,2)
  relu_leak   = 0.1
  padding     = 'same'
  in_dim = c(nn, nn, 1) #c(dim(x_train)[2:4])
  
  # Define a CNN model structure
  model <- keras_model_sequential() %>% 
    layer_conv_2d(filters = 128, kernel_size = kernel_size, padding = padding, input_shape = in_dim, activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_max_pooling_2d(pool_size, padding = padding)  %>%
    layer_conv_2d(filters = 128, kernel_size = kernel_size, padding = padding, activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_max_pooling_2d(pool_size, padding = padding)  %>%
    layer_conv_2d(filters = 16, kernel_size = kernel_size, padding = padding, activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_max_pooling_2d(pool_size, padding = padding)  %>%
    layer_flatten() %>%
    layer_dense(units = 4, activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak)  %>%
    layer_dense(units = 8, activation = "linear") %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_dense(units = 16, activation = "linear")  %>%
    layer_activation_leaky_relu(alpha = relu_leak) %>%
    layer_dense(units = 2) 
  
  summary(model)
  
  # set optimization parameters
  learning_rate = 0.01
  loss          = 'mse'    # mean squared error regression
  metrics       = 'mae'  # mean absolute error
  n_epochs      = 40#20
  batch_size    = 32
  
  # Compile model
  model %>% compile(
    optimizer_adam(lr = learning_rate), 
    loss = loss, 
    metrics = metrics)
  
  # Train model
  cnn_history <- model %>% fit(
    x_train, y_train,
    epochs = n_epochs,
    batch_size = batch_size,
    verbose = 2)
  
  y_pred_tmp = model %>% predict(x_test)
  y_pred = cbind(exp(y_pred_tmp[,1]), 2*exp(y_pred_tmp[,2])/(1 + exp(y_pred_tmp[,2])))
  
  return(y_pred = y_pred)
}

## ----------------
## Parameters transformations
## ----------------

orig2log = function(data, param){
  if(param =="nu"){
    out = log(data/(2 - data))
  }
  if(param =="range"){
    out = log(data)
  }
  return(out)
}
log2orig = function(data, param){
  if(param =="nu"){
    out = 2*exp(data)/(1+ exp(data))
  }
  if(param =="range"){
    out = exp(data)
  }
  return(out)
}