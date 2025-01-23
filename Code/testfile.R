
y = c(4.949,4.498,5.998,5.368,0.822,5.056)
sum((y-5)^2)

mu_new = -0.5954
mu_old = -0.7836

dnorm(mu_new, mean=-0.693, sd=sqrt(0.0125))
dnorm(mu_old, mean=mu_new, sd=sqrt(0.05))
dnorm(mu_old, mean=-0.693, sd=sqrt(0.0125))
dnorm(mu_new, mean=mu_old, sd=sqrt(0.05))


( dnorm(mu_new, mean=-0.693, sd=sqrt(0.0125)) * dnorm(mu_old, mean=mu_new, sd=sqrt(0.05)) ) /
  ( dnorm(mu_old, mean=-0.693, sd=sqrt(0.0125)) * dnorm(mu_new, mean=mu_old, sd=sqrt(0.05)) )


dinvgamma2 <- function(x, alpha, beta){
  beta^alpha / gamma(alpha) * (1/x)^(alpha+1) * exp(-beta / x)
}

theta_new = 0.9469
theta_old = 0.6030

( dinvgamma2(theta_new, alpha=7.63, beta=8.62) * dinvgamma2(theta_old, alpha=7.63, beta=8.62) ) /
  ( dinvgamma2(theta_old, alpha=7.63, beta=8.62) * dinvgamma2(theta_new, alpha=7.63, beta=8.62) )
