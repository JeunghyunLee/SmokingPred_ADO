data {
  int<lower=1> T;
  real<lower=0> r_ll[T];
  real<lower=0> r_ss[T];
  real<lower=0> t_ll[T];
  real<lower=0> t_ss[T];
  int<lower=0,upper=1> choice[T];
}

// Declare all parameters as vectors for vectorizing
parameters {
  real<lower=-10,upper=2> logk;
  real<lower=-4,upper=4> logtau;
}

model {
  logk ~ uniform(-10, 2);
  logtau ~ uniform(-4, 4);

  for (t in 1:T) {
    real sv_ll = r_ll[t] / (1 + exp(logk) * t_ll[t]);
    real sv_ss = r_ss[t] / (1 + exp(logk) * t_ss[t]);
    target += bernoulli_logit_lpmf(choice[t] | exp(logtau) * (sv_ll - sv_ss));
  }
}
