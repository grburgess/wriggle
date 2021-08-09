real partial_log_like( vector observed_counts,
                       vector background_counts,
                       vector background_errors,
                       vector source_spectrum,
                       int[] idx_background_zero,
                       int[] idx_background_nonzero,
                       int[] mask,
                       matrix rsp,
                       vector ene_width,
                       real exposure,
                       int N_ene, int N_chan) {


  vector[N_ene] integral_flux = source_spectrum .* ene_width;

  vector[N_chan] predicted_counts =  (rsp * integral_flux)  * exposure;

  return sum(pgstat(observed_counts[mask],
                    background_counts[mask],
                    background_errors[mask],
                    predicted_counts[mask],
                    idx_background_zero,
                    idx_background_nonzero));

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
