library(SpatialExtremes)

source("Code_MaxStable/auxiliary_functions.R")

# set variables 

n.rep.test = 50
nn = 25 
n.size = 20
x <- y <- seq(0, n.size, length = nn)
coord <- expand.grid(x, y) 

range_test = c(0.50, 0.75, 1.00, 1.50)
smooth_test = c(0.80, 1.05, 1.30, 1.55)

data_y = cbind(expand.grid(range_test, smooth_test)$Var1,
               expand.grid(range_test, smooth_test)$Var2)

# simulate testing data and plotting

data_x_test = simu_samp(data_y = data_y, n.rep.test = n.rep.test, 
                 nn = nn , coord = coord, cov.type="brown")

x <- y <- seq(0, n.size, length = nn)
rep = 2
par(mar=c(2, 3, 1, 1.2), mfrow=c(4,4), oma = c(1, 1, 0.5, 0.2))

for(ii in 1:16){
  data_x_mat <- matrix(data_x_test[ii,rep,,], nn, nn)
  
  if (sum(ii == c(1,5,9)) > 0){
    image(x, y, data_x_mat, font.main = 5,
          xaxt="n",
          cex.axis = 2,
          col = hcl.colors(12, "YlOrRd", rev = TRUE))
    
  } else if (sum(ii == c(14,15,16)) > 0){
    image(x, y, data_x_mat, font.main = 5,
          yaxt="n",
          cex.axis = 2,
          col = hcl.colors(12, "YlOrRd", rev = TRUE))
    
  } else if (ii == 13){
    image(x, y, data_x_mat, font.main = 5, 
          cex.axis = 2,
          col = hcl.colors(12, "YlOrRd", rev = TRUE)) 
  } else 
    image(x, y, data_x_mat, font.main = 5, 
          yaxt="n", xaxt="n",
          col = hcl.colors(12, "YlOrRd", rev = TRUE))
  
}

# change data format for the R-package

coord_mat = cbind(coord$Var1, coord$Var2) 
n.test = nrow(data_y)
data_x_array = array(NA, c(n.rep.test, n.test, nn^2))
for(im in 1:n.test){
  data_x_array[,im, ] = c(data_x_test[im, , ,])
}

## Initialize the parameters by giving the optimizer multiple random starting pairs around the actual values

weights <- as.numeric(distance(coord_mat) < 1) # Change '1' to higher values to include more neighbours
ni = 20
test.ini = array(NA, c(n.rep.test, n.test, ni))
s.s = s.r = matrix(NA, n.test, ni)

for(ipred in 1:n.test){
  for(ini in 1:ni){
    s.r[ipred,ini] = data_y[ipred,1] + rnorm(1, 0, data_y[ipred,1]/2) 
    if(s.r[ipred,ini] < 0) s.r[ipred,ini] = 0.1
    s.s[ipred,ini] = data_y[ipred,2] + rnorm(1, 0, 0.2)
    if(s.s[ipred,ini] < 0) s.s[ipred,ini] = 0.1
    if(s.s[ipred,ini] > 2) s.s[ipred,ini] = 0.1
    
      test.ini[,ipred,ini] = fit_BR(data = t(data_x_array[,ipred, ]),
                                   coords = coord_mat, 
                                   dist_weig = weights, 
                                   n.rep.test = n.rep.test, 
                                   s.init = s.s[ipred,ini],
                                   r.init = s.r[ipred,ini])$fit.like
    print(ini)
  }
  print(ipred)
}

## Run the full optimization from the five pairs with the highest pairwise likelihood among the random pairs as the starting point

n_ipl = 5
y_pred_pl = array(NA, c(n_ipl, n.rep.test, n.test, 2))

for(ipl in 1:n_ipl){
  for(ipred in 1:n.test){
    for(irep in 1:n.rep.test){
      init.opt.ind = order(test.ini[irep, ipred, ])[ipl]
      y_pred_pl[ipl, irep, ipred, ] = fitmaxstab(t(data_x_array[irep,ipred, ]), coord_mat, "brown", 
                                                 weights = weights, 
                                                 method = "L-BFGS-B",
                                                 start = list(range = s.r[ipred, init.opt.ind], 
                                                              smooth = s.s[ipred, init.opt.ind]))$param[1:2] 
    }
    print(ipred)
  }
  print(ipl)
}


# Simulate training data for the CNN in a large neighborhood of the truth

n_train = 2000
range = runif(n_train, 0.1, 3) 
smooth = runif(n_train, 0.5, 1.9)
params_mat_all = cbind(range, smooth)

data_x <- simu_samp(data_y = params_mat_all, n.rep.test = 1, 
                    nn = nn , coord = coord, cov.type="brown")


# Train the CNN and obtain predictions

y_pred_cnn = array(NA, c(n.rep.test, nrow(data_y), 2))

for(par in 1: nrow(data_y)){
  y_pred_cnn[,par,] = fit_pred_CNN(data_x_train = data_x, 
                            data_y_train = params_mat_all, 
                            data_x_pred = data_x_test[par,,,])
}

  
# Scatterplots 

ipl = 1

zlim_range_trans = round(range(c(orig2log(data = y_pred_pl[ipl,,,1], param = "range"), 
                                 orig2log(data = y_pred_cnn[,,1], param = "range"))), 3)
zlim_range_orig = round(range(c(y_pred_pl[ipl,,,1], y_pred_cnn[,,1])), 3)

zlim_nu_trans = round(range(c(orig2log(data = y_pred_pl[ipl,,,2], param = "nu"), 
                              orig2log(data = y_pred_cnn[,,2], param = "nu"))), 3)
zlim_nu_orig = round(range(c(y_pred_pl[ipl,,,2], y_pred_cnn[,,2])), 3)

par(mar=c(4, 4, 0.2, 0.2), mfrow=c(4,4), oma = c(0.2, 0.2, 1.5, 1.2))

for(i in 1:nrow(data_y)){
  
  range_transf_pl = orig2log(data = y_pred_pl[ipl,,i,1], param = "range")
  nu_transf_pl = orig2log(data = y_pred_pl[ipl,,i,2], param = "nu")
  
  range_transf_cnn = orig2log(data = y_pred_cnn[,i,1], param = "range")
  nu_transf_cnn = orig2log(data = y_pred_cnn[,i,2], param = "nu")
  
  range_trans_true = orig2log(data = data_y[i,1], param = "range")
  nu_trans_true = orig2log(data = data_y[i,2], param = "nu")
  
  if (sum(i == c(1,5,9)) > 0){
    
    plot(range_transf_pl, nu_transf_pl, 
         pch=20, col=2, lwd=3, #axes = F,
         mgp=c(2.5,1,0), 
         cex.lab = 3,
         xlim = zlim_range_trans,
         ylim= zlim_nu_trans,
         xaxt="n", yaxt="n",
         xlab = "", ylab=expression(nu))
    
    points(range_transf_cnn, nu_transf_cnn,  
           pch=3, lwd=2, col=3)
    
    points(range_trans_true, nu_trans_true, 
           pch=4, lwd=3, cex=3)
    
    axis(2, cex.axis=2.5, 
         labels = round(seq(0, zlim_nu_orig[2], by=0.05), 2),
         at = round(orig2log(data = seq(0, zlim_nu_orig[2], by=0.05), 
                             param = "nu"), 2))
    
  } else if (sum(i == c(14,15,16)) > 0){
    
    plot(range_transf_pl, nu_transf_pl, 
         pch=20, col=2, lwd=3, #axes = F,
         mgp=c(3,1,0), cex.lab = 3,
         xlim = zlim_range_trans,
         ylim= zlim_nu_trans,
         xaxt="n", yaxt="n",
         xlab = expression(lambda), ylab="")
    
    points(range_transf_cnn, nu_transf_cnn,  
           pch=3, lwd=2, col=3)
    
    points(range_trans_true, nu_trans_true, 
           pch=4, lwd=3, cex=3)
    
    axis(1, cex.axis=2.5,
         labels = round(seq(0.1, 2.4, by=0.1), 2),
         at = round(orig2log(data =seq(0.1, 2.4, by=0.1), 
                             param = "range"), 2))
    
  } else if (i == 13){
    
    plot(range_transf_pl, nu_transf_pl, 
         pch=20, col=2, lwd=3, #axes = F,
         mgp=c(2.5,1,0), cex.lab = 3,
         xlim = zlim_range_trans,
         ylim= zlim_nu_trans,
         xaxt="n", yaxt="n",
         xlab = expression(lambda), ylab=expression(nu))
    
    points(range_transf_cnn, nu_transf_cnn,  
           pch=3, lwd=2, col=3)
    
    points(range_trans_true, nu_trans_true, 
           pch=4, lwd=3, cex=3)
    
    axis(1, cex.axis=2.5,
         labels = round(seq(0.1, 2.4, by=0.1), 2),
         at = round(orig2log(data =seq(0.1, 2.4, by=0.1), 
                             param = "range"), 2))
    
    axis(2, cex.axis=2.5, 
         labels = round(seq(0, zlim_nu_orig[2], by=0.05), 2),
         at = round(orig2log(data = seq(0, zlim_nu_orig[2], by=0.05), 
                             param = "nu"), 2))
    
  } else{ 
    
    plot(range_transf_pl, nu_transf_pl, 
         pch=20, col=2, lwd=3, #axes = F,
         mgp=c(2.8,1,0),
         xlim = zlim_range_trans,
         ylim= zlim_nu_trans,
         yaxt="n", xaxt="n",
         xlab = "", ylab = "")
    
    points(range_transf_cnn, nu_transf_cnn,  
           pch=3, lwd=2, col=3)
    
    points(range_trans_true, nu_trans_true, 
           pch=4, lwd=3, cex=3)
  }
  
}


