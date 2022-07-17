functions {
  // function to efficiently compute part of the prediction for 'not very tall'
  vector not_very_exh_exponent(vector x,vector ls,int localN){
    vector[localN] logPhi;
    for(i in 1:localN){
      logPhi[i] = std_normal_lcdf(x[i]-ls[i]);
    }
    return square(ls)/2.0 +fma(- ls,x,logPhi);
  }
  // function to compute the posterior exhaustivity probability given 'not very tall' from pre-computed utilities and alpha,costs
  real posterior_exh_tall(
    real alpha,
    vector costs,
    data int K,
    data real prior_mean,
    data real prior_sd,
    data vector U1_null_tall,
    data vector U1_tall_pos,
    data vector U1_not_tall,
    data vector U1_very_tall,
    data vector U1_not_very_tall_lit,
    data vector U1_not_very_tall_exh,
    data vector log_prior,
    data array[,] int ii
  ) {
    // logS1
    // From now on, we omit the 'log' prefix before S1 and L1
    vector[K] S1_null_tall;
    vector[K] S1_tall_pos;
    vector[K] S1_not_tall;
    vector[K] S1_very_tall;
    vector[K] S1_not_very_tall_lit;
    vector[K] S1_not_very_tall_exh;
    vector[K] S1_const;
    // logL1 (only computed for 'not very tall' since we just need the posterior P(EXH|not very tall))
    vector[K] L1_lit_not_very_tall;
    vector[K] L1_exh_not_very_tall;
    // normalizing constants for L1 (log scale):
    real const_lit_not_very_tall;
    real const_exh_not_very_tall;
    // posterior parameters on logit scale
    real post_exh;
    // First compute unnormed logS1:
    S1_null_tall = alpha * (U1_null_tall + costs[1]);
    S1_tall_pos = alpha * (U1_tall_pos);
    S1_not_tall = alpha * (U1_not_tall-costs[2]);
    S1_very_tall = alpha * (U1_very_tall-costs[3]);
    S1_not_very_tall_lit = alpha * (U1_not_very_tall_lit-costs[2]-costs[3]);
    S1_not_very_tall_exh = alpha * (U1_not_very_tall_exh-costs[2]-costs[3]);
    // Compute the normalizing constant and normalize:
    for (k in 1:K) {
      S1_const[k] = log_sum_exp({
        S1_null_tall[k],
        S1_tall_pos[k],
        S1_not_tall[k],
        S1_very_tall[k],
        S1_not_very_tall_lit[k],
        S1_not_very_tall_exh[k]
      });
    }
    S1_null_tall = S1_null_tall-S1_const;
    S1_tall_pos = S1_tall_pos-S1_const;
    S1_not_tall = S1_not_tall-S1_const;
    S1_very_tall = S1_very_tall-S1_const;
    S1_not_very_tall_lit = S1_not_very_tall_lit-S1_const;
    S1_not_very_tall_exh = S1_not_very_tall_exh-S1_const;
    // Compute unnormed logL1:
    L1_lit_not_very_tall = S1_not_very_tall_lit + log_prior;
    L1_exh_not_very_tall = S1_not_very_tall_exh + log_prior;
    // Compute normalizing constant with Simpson's composite 3/8 rule (on log-scale):
    // We have to append everything first
    const_lit_not_very_tall = log_sum_exp({log_sum_exp(L1_lit_not_very_tall[ii[1]]),log_sum_exp(L1_lit_not_very_tall[ii[2]]+log(3)), log_sum_exp(L1_lit_not_very_tall[ii[3]]+log(3)), log_sum_exp(L1_lit_not_very_tall[ii[4]])});
    const_exh_not_very_tall = log_sum_exp({log_sum_exp(L1_exh_not_very_tall[ii[1]]),log_sum_exp(L1_exh_not_very_tall[ii[2]]+log(3)), log_sum_exp(L1_exh_not_very_tall[ii[3]]+log(3)), log_sum_exp(L1_exh_not_very_tall[ii[4]])});
    // This gives us the posterior on EXH on the logit scale:
    post_exh = const_exh_not_very_tall-const_lit_not_very_tall;
    return post_exh;
  }
  // function to compute the posterior parse probabilities for not late/not very late from pre-computed utilities and alpha,costs
  vector posterior_parses_late(
    real alpha,
    vector costs,
    data int K,
    data vector times, // we need the times to distinguish between d>0 and d<=0 below
    data real prior_mean,
    data real prior_sd,
    data vector U1_null_late,
    data vector U1_late_min,
    data vector U1_late_pos,
    data vector U1_not_late_min,
    data vector U1_not_late_pos,
    data vector U1_very_late,
    data vector U1_not_very_late_lit,
    data vector U1_not_very_late_exhpos,
    data vector log_prior,
    data array[,] int ii
  ) {
    // With 'late', some (u,i) pairs have -Inf utility when d is either positive or negative,
    // and this creates issues when computing the gradient for alpha.
    // We solve this by using vectors of 1 and alpha, where 1 replaces alpha in places where utility is known to be -Inf
    // (see below for how we use these vectors)
    vector[K] alpha_pos;
    vector[K] alpha_neg;
    // logS1
    vector[K] S1_null_late;
    vector[K] S1_late_min;
    vector[K] S1_late_pos;
    vector[K] S1_not_late_min;
    vector[K] S1_not_late_pos;
    vector[K] S1_very_late;
    vector[K] S1_not_very_late_lit;
    vector[K] S1_not_very_late_exhpos;
    vector[K] S1_const;
    // logL1 (computed for 'not late' to get posterior on MIN and 'not very late' for posteriors on EXH_MIN and EXH_POS)
    vector[K] L1_min_late;
    vector[K] L1_pos_late;
    vector[K] L1_lit_not_very_late;
    vector[K] L1_exhpos_not_very_late;
    // normalizing constants for L1 (log scale):
    real const_min_late;
    real const_pos_late;
    real const_lit_not_very_late;
    real const_exhpos_not_very_late;
    // posterior parameters on logit scale
    real post_logit_min;
    real post_logit_exhpos;
    // real post_exh;
    for (k in 1:K) {
      alpha_pos[k] = (times[k] > 0 ? alpha : 1);
      alpha_neg[k] = (times[k] > 0 ? 1 : alpha);
    }
    // First compute unnormed logS1:
    S1_null_late = alpha * (U1_null_late + costs[1]);
    S1_late_min = alpha_pos .* (U1_late_min); // late has utility -Inf if d negative or null
    S1_late_pos = alpha_pos .* (U1_late_pos);
    S1_not_late_min = alpha_neg .* (U1_not_late_min-costs[2]); // conversely, 'not MIN late' has -Inf utility if d>0
    S1_not_late_pos = alpha * (U1_not_late_pos-costs[2]); // by contrast, 'no POS late' doesn't strictly exclude any d, so its utility remains finite everywhere
    S1_very_late = alpha_pos .* (U1_very_late-costs[3]);
    S1_not_very_late_lit = alpha * (U1_not_very_late_lit-costs[2]-costs[3]);
    S1_not_very_late_exhpos = alpha_pos .* (U1_not_very_late_exhpos-costs[2]-costs[3]);
    // Compute the normalizing constant and normalize:
    for (k in 1:K) {
      S1_const[k] = log_sum_exp({
        S1_null_late[k],
        S1_late_min[k],
        S1_late_pos[k],
        S1_not_late_min[k],
        S1_not_late_pos[k],
        S1_very_late[k],
        S1_not_very_late_lit[k],
        S1_not_very_late_exhpos[k]
      });
    }
    S1_null_late = S1_null_late-S1_const;
    S1_late_min = S1_late_min-S1_const;
    S1_late_pos = S1_late_pos-S1_const;
    S1_not_late_min = S1_not_late_min-S1_const;
    S1_not_late_pos = S1_not_late_pos-S1_const;
    S1_very_late = S1_very_late-S1_const;
    S1_not_very_late_lit = S1_not_very_late_lit-S1_const;
    S1_not_very_late_exhpos = S1_not_very_late_exhpos-S1_const;
    // Compute unnormed logL1:
    L1_min_late = S1_late_min + log_prior;
    L1_pos_late = S1_late_pos + log_prior;
    L1_lit_not_very_late = S1_not_very_late_lit + log_prior;
    L1_exhpos_not_very_late = S1_not_very_late_exhpos + log_prior;
    // Compute normalizing constant to marginalize over d with Simpson's composite 3/8 rule (on log-scale):
    // We have to append everything first (alternatively, we could call log_sum_exp three times, but I doubt it's more efficient)
    const_min_late = log_sum_exp({log_sum_exp(L1_min_late[ii[1]]),log_sum_exp(L1_min_late[ii[2]]+log(3)), log_sum_exp(L1_min_late[ii[3]]+log(3)),log_sum_exp(L1_min_late[ii[4]])});
    const_pos_late = log_sum_exp({log_sum_exp(L1_pos_late[ii[1]]),log_sum_exp(L1_pos_late[ii[2]]+log(3)), log_sum_exp(L1_pos_late[ii[3]]+log(3)),log_sum_exp(L1_pos_late[ii[4]])});
    const_lit_not_very_late = log_sum_exp({log_sum_exp(L1_lit_not_very_late[ii[1]]),log_sum_exp(L1_lit_not_very_late[ii[2]]+log(3)), log_sum_exp(L1_lit_not_very_late[ii[3]]+log(3)),log_sum_exp(L1_lit_not_very_late[ii[4]])});
    const_exhpos_not_very_late = log_sum_exp({log_sum_exp(L1_exhpos_not_very_late[ii[1]]),log_sum_exp(L1_exhpos_not_very_late[ii[2]]+log(3)), log_sum_exp(L1_exhpos_not_very_late[ii[3]]+log(3)),log_sum_exp(L1_exhpos_not_very_late[ii[4]])});
    // Compute posteriors on MIN and EXH_POS and EXH_MIN (alt: EXH), on logit scale:
    post_logit_min = const_min_late-const_pos_late;
    post_logit_exhpos = const_exhpos_not_very_late-const_lit_not_very_late;
    return [post_logit_min,post_logit_exhpos]';
  }
}


data {
  int N;                            // number of observations
  int S;                            // number of participants
  vector[N] unit;                          // unit
  array[N] int<upper=S> subject;               // subject id
  array[N] int<lower=0,upper=1> adj;           // adjective (0 for tall, 1 for late)
  array[N] int<lower=0,upper=1> adv;           // adverb (0 for bare adj, 1 for very adj)
  vector<lower=0,upper=1>[N] y;                // response in [0,1]
  array[S] int<lower=0,upper=1> subj_adj;      // adjective by participant (0 for tall, 1 for late)
  // prior parameters
  real tall_prior_mean;
  real tall_prior_sd;
  real late_prior_mean;
  real late_prior_sd;
  // parameters of the Utility sampling:
  int K;                            // number of sampled degrees, must be of the form 3n+1, same for tall and late
  vector[K] heights;                // sampled height degrees
  vector[K] times;                  // sampled time degrees
  // Precomputed utilies for tall (without cost):
  vector[K] U1_null_tall;
  vector[K] U1_tall_pos;
  vector[K] U1_not_tall;
  vector[K] U1_very_tall;
  vector[K] U1_not_very_tall_lit;
  vector[K] U1_not_very_tall_exh;
  // Precomputed utilies for late (without cost):
  vector[K] U1_null_late;
  vector[K] U1_late_min;
  vector[K] U1_late_pos;
  vector[K] U1_not_late_min;
  vector[K] U1_not_late_pos;
  vector[K] U1_very_late;
  vector[K] U1_not_very_late_lit;
  vector[K] U1_not_very_late_exhpos;
  // Parameters fitted to individual participants in the positive case:
  vector[S] mu_theta_tall;
  vector<lower=0>[S] sigma_theta_tall;
  vector<lower=0>[S] lambda_delta_tall;
  vector<lower=0>[S] lambda_theta_late;
  vector<lower=0>[S] lambda_delta_late;
}

transformed data {
  // log priors:
  vector[K] log_prior_tall;
  vector[K] log_prior_late;
  array[S] row_vector[N] subject_index;    // used in the computation of log-lik for LOO
  // indices arrays for the integrals:
  array[4,(K-1)%/%3] int ii;
  for (k in 1:K) {
    log_prior_late[k] = normal_lpdf(times[k]|late_prior_mean,late_prior_sd);
    log_prior_tall[k] = normal_lpdf(heights[k]|tall_prior_mean,tall_prior_sd);
  }
  for (i in 1:4) {
    for (j in 1:((K-1)%/%3)) {
      ii[i,j] = 3*(j-1)+i;
    }
  }
  for (s in 1:S){
    for (n in 1:N){
      subject_index[s][n] = (subject[n]==s);
    }
  }
}

parameters {
  real m_alpha;              // mean log rationality parameter
  real m_cost_sen;      // mean cost for tall/late
  real<lower=0> cost_neg;           // mean cost for negation
  real<lower=0> cost_very;          // mean cost for very
  matrix[2,S] z_v;              // normed RE on alpha and costs
  vector<lower=0>[2] s_v;       // sd of RE on alpha and costs
  //cholesky_factor_corr[4] L_v;  // alpha and cost RE correlation matrix
  real<lower=0> eps;            // noise parameter
}


transformed parameters {
  // random effects:
  matrix[2,S] v;
  // parameters by participant:
  vector<lower=0>[S] alpha; // rationality
  vector[S] cost_sen; // cost of the affirmative bare adjective sentence
  array[S] real post_logit_exhpos; // posterior probability of the exp_pos parse for 'not very adj' on logit scale
  array[S] real post_logit_min; // posterior probability of the min parse for 'not adj' on logit scale
  vector[N] pred; // vector of pointwise predictions
  { // open block so that everything else is treated as local variable and not saved
  vector[N] pred_not_tall; // vector of pointwise predictions for not_tall
  vector[N] pred_not_very_tall;
  vector[N] pred_not_late; // vector of pointwise predictions for not_late (only valid for positive times)
  vector[N] pred_not_very_late; // vector of pointwise predictions for not_very_late (only valid for positive times)
  vector[S] ls_tall; // product of lambda_delta_tall, sigma_theta_tall
  vector[S] coeff_late; // 1/(lambda_delta_late - lambda_theta_late)
  vector[N] lccdf_tall; // normal_lccdf(unit|mu_tall,sigma_tall)
  vector[N] nvee; // store exponent for not very tall
  vector[S] post_lit; // posterior probability of lit interpretation (used for not very late)
  vector[S] lambda_mix_vector;
  // Compute RE:
  //v = diag_pre_multiply(s_v, L_v) * z_v;
  v = diag_matrix(s_v) * z_v;
  for (s in 1:S) {
    // transform to individual participants parameters:
    alpha[s] = exp(m_alpha + v[1,s]);
    cost_sen[s] = m_cost_sen + v[2,s];
    // cost_neg[s] = exp(m_cost_neg  + v[3,s]);
    // cost_very[s] = exp(m_cost_very  + v[4,s]);
    // compute posteriors on parses predicted by the RSA-SvI model:
    if (subj_adj[s]) { // late case
      vector[2] loc_post_param; // local variable to store posterior parameters
      loc_post_param = posterior_parses_late(
        alpha[s],[cost_sen[s],cost_neg,cost_very]',K, times, late_prior_mean,late_prior_sd,U1_null_late,
        U1_late_min,U1_late_pos,U1_not_late_min,U1_not_late_pos,U1_very_late,U1_not_very_late_lit,
        U1_not_very_late_exhpos,log_prior_late,ii
      );
      post_logit_min[s] = loc_post_param[1];
      post_logit_exhpos[s] = loc_post_param[2];
    } else { // tall case
      post_logit_min[s] = 0; // should be negative_infinity(), but let's keep it at 0 since we won't use it anyway
      post_logit_exhpos[s] = posterior_exh_tall(
        alpha[s],[cost_sen[s],cost_neg,cost_very]',K, tall_prior_mean,tall_prior_sd,U1_null_tall,
      U1_tall_pos,U1_not_tall,U1_very_tall,U1_not_very_tall_lit,
      U1_not_very_tall_exh,log_prior_tall,ii
      );
    }
  }
  for (i in 1:N) {lccdf_tall[i] = normal_lccdf(unit[i]|mu_theta_tall[subject[i]],sigma_theta_tall[subject[i]]);}
  // compute predicted acceptability:
  pred_not_tall = exp(lccdf_tall);
  pred_not_very_tall = log1m_inv_logit(to_vector(post_logit_exhpos[subject]))+lccdf_tall; // temporary
  ls_tall = lambda_delta_tall.*sigma_theta_tall;
  nvee = not_very_exh_exponent((unit-mu_theta_tall[subject])./sigma_theta_tall[subject],ls_tall[subject],N);
  for (i in 1:N) {
    pred_not_very_tall[i] = exp(log_sum_exp(pred_not_very_tall[i],nvee[i]));
  }
  pred_not_late = exp(log1m_inv_logit(to_vector(post_logit_min[subject]))-lambda_theta_late[subject].*unit);
  coeff_late = inv(lambda_delta_late-lambda_theta_late); // sometimes <0, so we can't work with log
  for(s in 1:S){
    lambda_mix_vector[s] = log_mix(inv_logit(post_logit_exhpos[s]),log(lambda_theta_late[s]),log(lambda_delta_late[s]));
  }
  pred_not_very_late = coeff_late[subject].*(exp(lambda_mix_vector[subject]-lambda_theta_late[subject].*unit)
                       -lambda_theta_late[subject].*exp(-lambda_delta_late[subject].*unit));
  post_lit = exp(log1m_inv_logit(to_vector(post_logit_exhpos)));
  for (i in 1:N) {
    if (adj[i] && adv[i]) { // not very late
      pred[i] = (unit[i] > 0) ? pred_not_very_late[i] : post_lit[subject[i]];
    } else if (adj[i] && !adv[i]) { // not late
      pred[i] = (unit[i] > 0) ? pred_not_late[i] : 1;
    } else if (adv[i]) { // not very tall
      pred[i] =  pred_not_very_tall[i];
    } else { // not tall
      pred[i] = pred_not_tall[i];
    }
  }
  } // close block
}

model {
  // L_v ~ lkj_corr_cholesky(2.0);
  to_vector(z_v) ~ std_normal(); // normalized RE
  m_alpha ~ normal(0,1); // rather uninformative prior
  m_cost_sen ~ normal(0,2); // quite uninformative prior
  cost_neg ~ gamma(1.3,1.1); //  keep away from 0 to avoid divergences
  cost_very ~ gamma(1.3,1.5); 
  // cost_neg ~ lognormal(0.5,1); //  informative prior (negation expected to be costly)
  // cost_very ~ lognormal(-0.5,1); //  informative prior (very expected to have a small effect)
  s_v ~ gamma(1.3,4); // informative prior to make sure we only get small values but stay away from 0
  eps ~ gamma(2,5); // informative prior. We know we're going to be close to 0.2 if the fit is decent.
  y ~ normal(pred,eps);
}

generated quantities { // save log_lik by participant for LOO-CV
  array[S] real log_lik;
  {vector[N] pw_log_lik; // pointwise loglik declared as a local variable to save memory
  for (n in 1:N){
    pw_log_lik[n] = normal_lpdf(y[n]|pred[n],eps);
  }
  for (s in 1:S){
    log_lik[s] = subject_index[s]*pw_log_lik;
  }}
}



