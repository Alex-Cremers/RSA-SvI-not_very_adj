functions {
  // function to compute the exponent in 'very tall' a bit more efficiently:
  real very_exponent(real x,real ls){
    return ls^2/2 - ls*x +std_normal_lcdf(x-ls)-std_normal_lcdf(x);
  }
}

data {
  int N;                            // number of observations
  int S;                            // number of participants
  real unit[N];                     // unit
  int<upper=S> subject[N];          // subject id
  int<lower=0,upper=1> adv[N];      // adverb (0 for bare adj, 1 for very adj)
  real<lower=0,upper=1> y[N];       // response in [0,1]
}


parameters {
  real m_mu; // mean of mu (across participants)
  real m_sigma; // mean of sigma
  real m_lambda; // mean of lambda
  matrix[3,S] z_u;  // normed random effects by participant for mu/sigma/lambda
  cholesky_factor_corr[3] L_u; // subj correlation matrix
  real<lower=0> s_mu; // sd of mu (across participants)
  real<lower=0> s_sigma; // sd of sigma
  real<lower=0> s_lambda; // sd of lambda
  real<lower=0> eps; // noise parameter
}

transformed parameters {
  matrix[3,S] u;  //random effects
  vector[S] mu; // vector of mu for each participant
  vector<lower=0>[S] sigma; // sigma for each participant
  vector<lower=0>[S] lambda; // lambda for each participant
  vector[N] pred; // prediction for each data point
  u = diag_pre_multiply([s_mu,s_sigma,s_lambda]', L_u) * z_u;
  for (s in 1:S) {
    mu[s] = m_mu + u[1,s];
    sigma[s] = exp(m_sigma + u[2,s]);
    lambda[s] = exp(m_lambda + u[3,s]);
  }
  for (i in 1:N) {
    if (adv[i])
      pred[i] = normal_lcdf(unit[i]|mu[subject[i]],sigma[subject[i]]) + log1m_exp(very_exponent((unit[i]-mu[subject[i]])/sigma[subject[i]],lambda[subject[i]]*sigma[subject[i]]));
    else
      pred[i] = normal_lcdf(unit[i]|mu[subject[i]],sigma[subject[i]]);
  }
}

model {
  L_u ~ lkj_corr_cholesky(2.0);
  to_vector(z_u) ~ normal(0,1);
  m_mu ~ normal(0.5,1);
  m_sigma ~ normal(0,1);
  m_lambda ~ normal(0,3);
  s_mu ~ gamma(1.2,0.5);
  s_sigma ~ gamma(1.3,1.5);
  s_lambda ~ gamma(1.3,1.5);
  eps ~  gamma(2,5); 
  y ~ normal(exp(pred),eps);
}

