data {
  int N;                            // number of observations
  int S;                            // number of participants
  real unit[N];                     // unit
  int<upper=S> subject[N];          // subject id
  int<lower=0,upper=1> adv[N];       // adverb (0 for bare adj, 1 for very adj)
  real<lower=0,upper=1> y[N];       // response in [0,1]
}


parameters {
  real m_zeta;
  real m_lambda_theta;
  real m_lambda_delta;
  matrix[3,S] z_u;  // normed random effects by participant for zeta/lambda_theta/lambda_delta 
  cholesky_factor_corr[3] L_u; // subj correlation matrix
  vector<lower=0>[3] sigma_u; // subj sd for zeta, lambda_theta, lambda_delta
  real<lower=0> eps; // noise 
}

transformed parameters {
  matrix[3,S] u;  //random effects
  vector<lower=0,upper=1>[S] zeta; // vector of zeta for each participant
  vector<lower=0>[S] lambda_theta; // lambda_theta for each participant
  vector<lower=0>[S] lambda_delta; // lambda_delta for each participant
  u = diag_pre_multiply(sigma_u, L_u) * z_u;
  for (s in 1:S) {
    zeta[s] = inv_logit(m_zeta + u[1,s]);
    lambda_theta[s] = exp(m_lambda_theta + u[2,s]);
    lambda_delta[s] = exp(m_lambda_delta + u[3,s]);
  }
}

model {
  vector[N] pred;
  L_u ~ lkj_corr_cholesky(2.0);
  to_vector(z_u) ~ normal(0,1);
  sigma_u ~ gamma(1.2,1); // informative prior to make sure we don't get crazy high values
  eps ~ gamma(2,0.1); 
  m_zeta ~ normal(0,3);
  m_lambda_theta ~ normal(0,3);
  m_lambda_delta ~ normal(1,3);
  for (i in 1:N) {
    pred[i] = (unit[i] > 0) ? (adv[i] ? 
                1 - (lambda_delta[subject[i]]*exp(-lambda_theta[subject[i]]*unit[i])-lambda_theta[subject[i]]*exp(-lambda_delta[subject[i]]*unit[i]))/(lambda_delta[subject[i]] - lambda_theta[subject[i]]) :
                zeta[subject[i]] + (1-zeta[subject[i]])*(1-exp(-lambda_theta[subject[i]]*unit[i]))
              ) : 0;
  }
  y ~ normal(pred,eps);
}

