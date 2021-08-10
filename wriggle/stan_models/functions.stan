vector fold_counts(vector spectrum, vector ene_width, matrix rsp, real exposure){

  return (rsp * (spectrum .* ene_width)) * exposure;


}

real partial_log_like(int [] n_slice, int start, int end,

                      vector[] observed_counts,
                      vector[] background_counts,
                      vector[] background_errors,
                      vector[] source_spectrum,
                      int[,] idx_background_zero,
                      int[,] idx_background_nonzero,
                      int[,] mask,
                      matrix[] rsp,
                      vector[] ene_width,
                      real[] exposure,
                      int N_ene, int N_chan,
                      int[] det_type,
                      int[] N_channels_used,
                      int[] N_bkg_nonzero,
                      int[] N_bkg_zero,
                      real log_amplitude) {


  int slice_length = num_elements(n_slice);
  real loglike = 0.;



  for (n in 1:slice_length) {

    int m = n_slice[n]; // detector number

    loglike += sum(pgstat(observed_counts[m][mask[m][:N_channels_used[m]]],
                          background_counts[m][mask[m][:N_channels_used[m]]],
                          background_errors[m][mask[m][:N_channels_used[m]]],
                          fold_counts(10^log_amplitude * source_spectrum[det_type[m]], ene_width[det_type[m]], rsp[m], exposure[m])[mask[m][:N_channels_used[m]]],
                          idx_background_zero[m][:N_bkg_zero[m]],
                          idx_background_nonzero[m][:N_bkg_nonzero[m]]));


  }





  return loglike;

}

vector compute_model_spectrum(vector ene_center,
                              row_vector omega,
                              vector beta1,
                              vector beta2,
                              real scale,
                              int k,
                              int N_echan){

  matrix [N_echan, k] tw = ene_center * omega;
  //matrix [N,k] tw2 = ene_center * omega2;


  vector[N_echan] expected_counts_log = ((scale* cos(tw)) * beta1) + ((scale * sin(tw)) * beta2);

  return exp(expected_counts_log);

}
