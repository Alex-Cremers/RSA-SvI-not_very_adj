functions {
  real not_very_exponent(real x,real ls){
    return ls^2/2 - ls*x +std_normal_lcdf(x-ls);
  }
}

data {
  int N;                            // number of observations
  int S;                            // number of participants
  real unit[N];                     // unit
  int<upper=S> subject[N];          // subject id
  int<lower=0,upper=1> adj[N];      // adjective (0 for tall, 1 for late)
  int<lower=0,upper=1> adv[N];      // adverb (0 for bare adj, 1 for very adj)
  vector<lower=0,upper=1>[N] y;       // response in [0,1]
  int<lower=0,upper=1> subj_adj[S];      // adjective by participant (0 for tall, 1 for late)
  // Parameters fitted to individual participants in the positive case:
  vector[S] mu_theta_tall;
  vector<lower=0>[S] sigma_theta_tall;
  vector<lower=0>[S] lambda_delta_tall;
  vector<lower=0>[S] lambda_theta_late;
  vector<lower=0>[S] lambda_delta_late;
  vector<lower=0>[S] p_min_late;
}

transformed data {
  row_vector[N] subject_index[S]; // used to simplify the computation of log-lik for LOO
  for (s in 1:S){
    for (n in 1:N){
      subject_index[s][n] = (subject[n]==s);
    }
  }
}

parameters {
  real<lower=0> eps;            // noise parameter
}


transformed parameters {
  vector<lower=0,upper=1>[N] pred;
  for (i in 1:N) {
    if (adj[i] && adv[i]) {
      pred[i] = (unit[i] > 0) ? ((lambda_delta_late[subject[i]])*exp(-lambda_theta_late[subject[i]]*unit[i])-lambda_theta_late[subject[i]]*exp(-lambda_delta_late[subject[i]]*unit[i]))/(lambda_delta_late[subject[i]] - lambda_theta_late[subject[i]]) : 1;
    } else if (adj[i] && !adv[i]) {
      pred[i] = (unit[i] > 0) ? (1-p_min_late[subject[i]])*exp(-lambda_theta_late[subject[i]]*unit[i]) : 1;
    } else if (adv[i]) {
      pred[i] =  (1-normal_cdf(unit[i]|mu_theta_tall[subject[i]],sigma_theta_tall[subject[i]])) + 
        exp(not_very_exponent((unit[i]-mu_theta_tall[subject[i]])/sigma_theta_tall[subject[i]],lambda_delta_tall[subject[i]]*sigma_theta_tall[subject[i]]));
    } else {
      pred[i] = 1-normal_cdf(unit[i]|mu_theta_tall[subject[i]],sigma_theta_tall[subject[i]]);
    }
  }
}

model {
  eps ~ gamma(2,5); // informative prior. We know we're going to be close to 0.2 if the fit is decent.
  y ~ normal(pred,eps);
}

generated quantities {
  real log_lik[S];
  {vector[N] pw_log_lik; // declared as a local variable to save memory
  for (n in 1:N){
    pw_log_lik[n] = normal_lpdf(y[n]|pred[n],eps);
  }
  for (s in 1:S){
    log_lik[s] = subject_index[s]*pw_log_lik;
  }}
}


