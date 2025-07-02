functions {
  real subjective_value(real alpha, real beta, real p, real a, real v) {
    return (p - beta * a / 2) * pow(v, alpha);
  }
}

data {
  int<lower=1> T;
  real<lower=0,upper=1> p_var[T];
  real<lower=0,upper=1> a_var[T];
  real<lower=0> r_var[T];
  real<lower=0> r_fix[T];
  int<lower=0,upper=1> choice[T];
}

// Declare all parameters as vectors for vectorizing
parameters {
  real<lower=0,upper=5>  alpha;
  real<lower=-3,upper=3> beta;
  real<lower=-4,upper=4> loggamma;
}

model {
  alpha    ~ uniform(0, 5);
  beta     ~ uniform(-3, 3);
  loggamma ~ uniform(-4, 4);

  for (t in 1:T) {
    real u_fix = subjective_value(alpha, beta, 0.5, 0, r_fix[t]);
    real u_var = subjective_value(alpha, beta, p_var[t], a_var[t], r_var[t]);

    target += bernoulli_logit_lpmf(choice[t] | exp(loggamma) * (u_var - u_fix));
  }
}
