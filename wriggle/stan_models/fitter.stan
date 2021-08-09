functions {
#include pgstat.stan
#include functions.stan
}

data {



  int N_dets;
  int det_type [N_dets];
  int N_echan;
  int N_chan;



  vector[N_echan] ebounds_hi[N_dets];
  vector[N_echan] ebounds_lo[N_dets];



  vector[N_chan] observed_counts[N_dets];
  vector[N_chan] background_counts[N_dets];
  vector[N_chan] background_errors[N_dets];

  int idx_background_zero[N_dets, N_chan];
  int idx_background_nonzero[N_dets, N_chan];

  int N_bkg_zero[N_dets];
  int N_bkg_nonzero[N_dets];

  real exposure[N_dets];

  matrix[N_chan, N_echan] response[N_dets];



  int mask[N_dets, N_chan];
  int N_channels_used[N_dets];

  real max_range;

  int k;

}



transformed data {

  int N_total_channels = 0;

  vector[N_echan] ene_center[2];
  vector[N_echan] ene_width[2];


  // the photon side energies only need to
  // be stored once per detector type (bgo, nai)

  for (n in 1:N_dets){

    ene_center[det_type[n]] = 0.5 * (ebounds_lo[n] + ebounds_hi[n]);

    ene_width[det_type[n]] = (ebounds_hi[n] + ebounds_lo[n]);

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

  row_vector[k] omega_var; // this weird MC integration thing.


  real log_amplitude; // independent amplitude1 of LC 1; probably do not need right now...

  real<lower=0> raw_scale;

  real<lower=0, upper=1> range_raw;




}

transformed parameters {

  real scale = raw_scale * inv_sqrt(k);
  real range;
  real bw;
  row_vector[k] omega; 
  vector[N_echan] spectrum[2];



  range = range_raw * max_range;


  bw = inv(range);


  // non-center
  omega = omega_var * bw;

  for (n in 1:2){

    spectrum[n] = compute_model_spectrum(ene_center[n], omega, beta1, beta2, scale, k, N_echan);

  }



}


model {

  int grainsize = 1;

  // priors

  beta1 ~ std_normal();
  beta2 ~ std_normal();
  raw_scale ~ normal(1,1);
  range_raw ~ lognormal(0, .2);

  omega_var ~ std_normal();

  log_amplitude ~ normal(0,100);

  for (n in 1:N_dets){

    target += partial_log_like(observed_counts[n],
                               background_counts[n],
                               background_errors[n],
                               10^log_amplitude *  spectrum[det_type[n]],
                               idx_background_zero[n][:N_bkg_zero[n]],
                               idx_background_nonzero[n][:N_bkg_nonzero[n]],
                               mask[n][:N_channels_used[n]],
                               response[n],
                               ene_width[det_type[n]],
                               exposure[n],
                               N_echan,
                               N_chan);

  }



}
