functions {
#include pgstat.stan
#include functions.stan
}

data {



  int N_dets;
  array[N_dets]  int det_type ;
  int N_echan;
  int N_chan;



  array[N_dets] vector[N_echan] ebounds_hi;
  array[N_dets] vector[N_echan] ebounds_lo;



  array[N_dets] vector[N_chan] observed_counts;
  array[N_dets] vector[N_chan] background_counts;
  array[N_dets] vector[N_chan] background_errors;

  array[N_dets, N_chan] int idx_background_zero;
  array[N_dets, N_chan] int idx_background_nonzero;

  array[N_dets] int N_bkg_zero;
  array[N_dets] int N_bkg_nonzero;

  array[N_dets] real exposure;

  array[N_dets] matrix[N_chan, N_echan] response;



  array[N_dets, N_chan] int mask;
  array[N_dets] int N_channels_used;

  real max_range;

  int k;

}



transformed data {

  int N_total_channels = 0;

  array[2] vector[N_echan] ene_center;
  array[2] vector[N_echan] ene_width;
  array[N_dets] int all_N;

  // the photon side energies only need to
  // be stored once per detector type (bgo, nai)

  for (n in 1:N_dets){

    ene_center[det_type[n]] = 0.5 * (ebounds_lo[n] + ebounds_hi[n]);

    ene_width[det_type[n]] = (ebounds_hi[n] - ebounds_lo[n]);

    all_N[n] = n;


  }





  // vector[max_n_echan] ebounds_add[N_intervals, max(N_dets)];
  // vector[max_n_echan] ebounds_half[N_intervals, max(N_dets)];

  // int all_N[N_intervals];


  // precalculation of energy bounds

  // for (n in 1:N_intervals) {

  //   all_N[n] = n;

  //   for (m in 1:N_dets[n]) {
  //     ebounds_half[n, m, :N_echan[n, m]] = 0.5*(ebounds_hi[n, m, :N_echan[n, m]]+ebounds_lo[n, m, :N_echan[n, m]]);
  //     ebounds_add[n, m, :N_echan[n, m]] = (ebounds_hi[n, m, :N_echan[n, m]] - ebounds_lo[n, m, :N_echan[n, m]])/6.0;
  //     N_total_channels += N_channels_used[n,m];
  //   }


  // }




}



parameters {

  vector[k] beta1; // the amplitude along the cos basis
  vector[k] beta2; // the amplitude along the sin basis

  array[2] row_vector[k] omega_var; // this weird MC integration thing.


  real log_amplitude; // independent amplitude1 of LC 1; probably do not need right now...

  positive_ordered[2] raw_scale;

  real<lower=0, upper=1> range1_raw;
  real<lower=0, upper=range1_raw> range2_raw;





}

transformed parameters {

  real scale = raw_scale * inv_sqrt(k);
  array[2] real range;
  array[2] real bw;
  array[2] row_vector[k] omega;
  array[2] vector[N_echan] spectrum;

  range[1] = range1_raw * max_range;
  range[2] = range2_raw * max_range;

  bw = inv(range);

  // non-center

  omega[1] = omega_var[1] * bw[1];
  omega[2] = omega_var[2] * bw[2];

  
  for (n in 1:2) {

    spectrum[n] = compute_model_spectrum_ns(ene_center[n], omega[1], omega[2], beta1, beta2, scale1, scale2, k, N_echan);

  }



}


model {

  int grainsize = 1;

  // priors

  beta1 ~ std_normal();
  beta2 ~ std_normal();
  raw_scale ~ normal(1,1);
  range_raw ~ uniform(0,1);

  omega_var ~ std_normal();

  log_amplitude ~ normal(0,5);


  target += reduce_sum(partial_log_like,
                       all_N,
                       grainsize,
                       observed_counts,
                       background_counts,
                       background_errors,
                       spectrum,
                       idx_background_zero,
                       idx_background_nonzero,
                       mask,
                       response,
                       ene_width,
                       exposure,
                       N_echan,
                       N_chan,
                       det_type,
                       N_channels_used,
                       N_bkg_nonzero,
                       N_bkg_zero,
                       log_amplitude


                       );





}
generated quantities {
  // vector[N_chan] counts[N_dets];

  // for (n in 1:N_dets){

  //   counts[n] = fold_counts(spectrum[det_type[n]], ene_width[det_type[n]], response[n], exposure[n]);


  // }




}
