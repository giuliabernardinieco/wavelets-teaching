import numpy as np
import sys
import pycwt
import pandas as pd

# functions from the processing_wav.py file
from wavelets_wrapper_posh.processing_wav import pad, autoscales, recon, fourier_from_scales, icwt_fixed



# This function performs a full wavelet analysis on the input data file.
def run_full_wavelet_analysis(signal, dt=10000., mirror=True, period_min=None, period_max=None, wf='morlet', dj=0.1, om0=6, normmean=True, mirrormethod=1):
	"""
		Run a full continuous wavelet analysis on a single time-series signal.
		This function performs the following operations:

		1. **Signal preprocessing**
		* Optional mirroring of the signal to reduce edge effects.
		* Optional mean normalisation.

		2. **Fourier analysis**
		* Computes the FFT of the signal and the corresponding power spectrum.

		3. **Continuous Wavelet Transform (CWT)**
		* Computes the wavelet transform using the selected mother wavelet.
		* Calculates wavelet power as the squared magnitude of the coefficients and adjust power to dt.
		* Converts scales to Fourier-equivalent periods.

		4. **Inverse Wavelet Transform**
		* Reconstructs the original signal from all wavelet scales (ICWT).
		* Optionally reconstructs a **band-pass filtered signal** using only
			wavelet scales corresponding to a user-defined period range.

		## Parameters
		signal : array-like time series.

		dt : float, default=10000. Sampling interval of the signal.

		mirror : bool, default=True If True, the signal is mirrored to minimise edge effects before
		performing the wavelet transform.

		period_min : float, optional
		Minimum Fourier period used for band-pass ICWT reconstruction.

		period_max : float, optional
		Maximum Fourier period used for band-pass ICWT reconstruction.

		wf : str, default='morlet'
		Mother wavelet type. Supported options:
		- 'morlet'
		- 'dog'
		- 'paul'

		dj : float, default=0.1
		Scale resolution of the wavelet transform.

		om0 : float, default=6
		Central frequency parameter of the mother wavelet.

		normmean : bool, default=True
		If True, subtracts the mean from the signal before analysis.

		mirrormethod : int, default=1
		Method used to mirror the signal.

		## Returns

		xmirror_df : pandas.DataFrame
		Mean value of the mirrored signal.

		fft_df : pandas.DataFrame
		Fourier frequency spectrum with power and smoothed power.

		scales_df : pandas.DataFrame
		Table of wavelet scales, equivalent Fourier periods, and frequencies.

		icwt_df : pandas.DataFrame
		Signal reconstructed from the inverse continuous wavelet transform.

		icwt_part1_df : pandas.DataFrame
		Band-pass filtered reconstruction based on the selected period range
		(None if no period range is provided).

		wavelet_power_df : pandas.DataFrame
		Flattened table containing the wavelet power spectrum:
		scale index, time, power, period, frequency, and wavelet coefficients.

		coi : numpy.ndarray
		Cone of influence values for each time step.

		cwtX : tuple
		Raw output of the pycwt continuous wavelet transform.

		scales : numpy.ndarray
		Wavelet scale vector used in the transform.

		x : numpy.ndarray
		Processed signal used for the wavelet transform (after mirroring
		and mean normalization).
		"""

	# Initialise the dataframes that will be returned by the function  
	xmirror_df = None
	fft_df = None
	scales_df = None
	icwt_df = None
	icwt_part1_df = None
	wavelet_power_df = None
	
	# om0, central freq of wavelet  
	p = int(om0) #om0 (read as the greek letter and then 0) is the parameter that define the central frequency of the wavelet, 
 					#for morlet this is often set to 6 as this provides a good compromise between time and frequency resolution.
	if wf == 'morlet':
		fullwavelet = pycwt.wavelet.Morlet(p)
	elif wf == 'dog':
		fullwavelet = pycwt.wavelet.DOG(p)
	elif wf == 'paul':
		fullwavelet = pycwt.wavelet.Paul(p)
	else:
		sys.exit('Could not recognise desired mother wavelet, exiting...')

	## read in signal ##
	x = signal 
	x_orig = x 
	
	## mirror/flip and save mean of result ##
	#  In your notes you can find more text on why mirroring and normalisation are important  
	if mirror == True : # if mirror is true, then the signal is mirrored
		xmax = np.amax(x)

		if mirrormethod == 1:
			xstart = x[0]
			xend = x[-1]
			x_rev = x[::-1]
			x_rev_flip = np.multiply(x_rev,-1.) + xend*2.
			x_flip = np.multiply(x,-1.) + xstart*2
			x = np.concatenate((x_orig,x_rev_flip,x_flip,x_rev,x_orig,x_rev_flip,x_flip,x_rev),axis=0)
		if mirrormethod == 3:
			x_rev = x[::-1]
			x = np.concatenate((x,x_rev,x,x_rev,x,x_rev,x,x_rev),axis=0)
		if mirrormethod == 4:
			x_rev = x[::-1]
			x_rot = -x_rev
			x = np.concatenate((x, x_rot, x, x_rot, x, x_rot, x, x_rot), axis=0)
		if mirrormethod == 2:
			x_rev = x[::-1]
			x_flip = (x * -1.)
			x_rev_flip = (x_rev * -1.) 
			x = np.concatenate((x,x_rev,x_flip,x_rev_flip,x,x_rev,x_flip,x_rev_flip),axis=0)

		xmirrormean = np.mean(x)
		xmirror_df = pd.DataFrame({'mirror_mean': [xmirrormean]}) #<====== GB addition

		if normmean == True: # if normmean is true, then the mean is subtracted from the signal.
			x = x - xmirrormean
	else:
		if normmean == True: # in any case the signal is always demeaned  if normmean is true
			x = x - np.mean(x)

	# get len of the new mirrorer signal (or original signal if mirror is false) for later use  
	xlen = len(x)
	
	## calculate fft of signal ##
	fft_signal = np.fft.rfft(x) # calculate the real-valued Fast Fourier Transform (FFT) of the signal
	fft_freq = np.fft.rfftfreq(xlen, d=dt) # returns the Discrete Fourier Transform sample frequencies corresponding to the FFT result
							# xlen: Length of the input signal
							# d: Sample spacing (inverse of the sampling rate)
 

	fft_freq = fft_freq[1:int((xlen/2)+1)] # we are trimming the to exclude the zero frequency (the mean) ? - to be checked with G
	fft_signal = (2. / xlen) * abs(fft_signal[1:int(xlen/2)+1]) # not sure what this does ? Are these the amplitudes - to be checked with G
	fft_power  = abs(fft_signal[0:int(xlen/2)+1]) ** 2. # calculate the power of the FFT - to be checked with G

	fftave = np.convolve(fft_power, np.ones(5)/5, mode='same') # This line smooths the power spectrum using a moving average filter. 

	### saving np fft results separate to those done internally by pycwt (by pyfftw.interfaces.scipy_fftpack - see helpers.py from pycwt)
	fft_freq_np = fft_freq
	fft_power_np = fft_power
	fft_ave_np = fftave
	
	fft_df = pd.DataFrame({'frequency': fft_freq_np, 'power': fft_power_np, 'smoothed_power': fft_ave_np})  #<====== GB addition


	## calculate wavelet scales ##
	N=int(x.shape[0]) # N is the length of the signal (padded, but in this case it is not padded)

	# smallest scale to be used in the wavelet transform, set to twice the sampling interval (dt) 
 	# this is a common choice to ensure that the smallest scale captures features that are at least as large as the sampling interval, avoiding aliasing issues.  
	s0 = 2.*dt 
 
	# Number of scales for the padded signal
	j_full = (1/dj) * np.log(N*dt /s0) / np.log(2.)
	# remeber: dj = Scale resolution (spacing between scales), set at the beginning of the function
	

	## perform continuous wavelet transform ##
	cwtX = pycwt.cwt(x, dt, dj=dj, s0=s0, J=j_full, wavelet=fullwavelet) # perform the continuous wavelet transform on padded signal
	X = cwtX[0] # Wavelet coefficients (complex values representing the signal's features at different scales and positions)
	X_orig_wav_coeff = X.copy() # make a copy of the wavelet coefficients for later use (before scaling)
	sj = cwtX[1] # Scales used in the CWT
	coi = cwtX[3] # cone of influence
	power = (np.abs(X))**2. # Wavelet power spectrum (magnitude squared of the wavelet coefficients)
	# multiplying for the np.sqrt(dt) allows for taking into account the discretisation and the sampling interval, 
 	# ensuring that the power is correctly scaled for the continuous wavelet transform. 
 
 	# renaming scales for easier handling later on, but they are the same as sj :-)
	scales = sj
	
	## convert scales to fourier periods 
	period = fourier_from_scales(scales,wf=wf,p=p)
	scale_len = len(scales)
	scales_df = pd.DataFrame({'index': range(scale_len), 'scale': scales, 'period': period, 'frequency': 1./period}) #<====== GB addition
 
	## Gabor limit check ##
		# The Gabor limit (or Gabor-Heisenberg limit) is a principle in signal processing that states there is a trade-off between
  		# time and frequency resolution. 
		# It essentially means that you cannot simultaneously achieve high resolution in both time and frequency domains.
		# The inverse wavelet transform is used to reconstruct the original signal from its wavelet coefficients. 
		# This step verifies that the correct scales have been used and checks the accuracy of the wavelet transform.
	x_array = dt*np.arange(1, len(x)+1)
	freqs = 1/period
	s1 = np.std(x_array)
	s2 = np.std(freqs)
	gtest = s1*s2
	if gtest < 1/(4*np.pi):
		print('\nWarning, signal spacing/frequency choice may not conform to Gabor limit...\n')

	# perform inverse transform (to check correct scales have been used)...
	## NOTE there was an error in previous versions of pycwt - check wavelet.py line 170 in icwt - sj should be square rooted on bottom of iW = ...
	## AND brackets should be added... should read:
	# iW = (dj * np.sqrt(dt) / (wavelet.cdelta * wavelet.psi(0)) *
	#          (np.real(W) / np.sqrt(sj)).sum(axis=0))
	# and then will work fine
	# INCLUDES calculation of recon factor (from empirical cdelta) unlike mlpy
	
	x_icwt = icwt_fixed(X_orig_wav_coeff, sj, dt, dj=dj, wavelet=fullwavelet).real
	icwt_df = pd.DataFrame({'inverse_cwt': x_icwt}) #<====== GB addition
	
	# calculate mean squared error of icwt... not used currently
	# diff = np.sqrt((x_pad - x_icwt)**2.)
	# diffmean = np.mean(diff)
 
	if period_min is not None and period_max is None:
		print('Please specify two cut-off periods to calculate ICWTs...') 
    #Bandpass filter the signal using the inverse CWT
	elif period_min is not None and period_max is not None:

		#convert to  np array for easier handling 
		period = np.array(period)
		scales = np.array(scales)
  
		# find the indices of the scales that correspond to the desired period range
		idx_low = np.argmin(np.abs(period - period_min))
		idx_high = np.argmin(np.abs(period - period_max))
		
  		# making sure the order of the indeces is correct for slicing the arrays
		i1 = min(idx_low, idx_high)
		i2 = max(idx_low, idx_high)

		# Extract wavelet coefficients for the desired scale ranges
		X_cut_window = X[i1:i2+1, :]
		scales_window = scales[i1:i2+1]

		# perform IWCT with only those scales and wavelet coeff
		x_icwt_window = icwt_fixed(
			X_cut_window,
			scales_window,
			dt,
			dj=dj,
			wavelet=fullwavelet
		).real

 		# save as df
		icwt_part1_df = pd.DataFrame({
			"inverse_cwt_part1": x_icwt_window
		})
	else:
		print('Not calculating bandpass filter')
	
	# this is the most important result that you want to save, vectorised for faster execution
	time = np.arange(X.shape[1]) * dt
	n_scales = len(scales)
	n_time = len(time)

	wavelet_power_df = pd.DataFrame({
		"row": np.repeat(np.arange(n_scales), n_time),
		"time": np.tile(time, n_scales),
		"power": power.flatten(),
		"period": np.repeat(period, n_time),
		"frequency": np.repeat(1./period, n_time),
		"rectified_power": (power / scales[:, None]).flatten()
	})
 
	return (xmirror_df, fft_df, scales_df, icwt_df, icwt_part1_df, wavelet_power_df, coi, cwtX, scales, x)

def run_double_wavelet_analysis(signal1, signal2, dt=10000., mirror=True, period_min=None, period_max=None, wf='morlet', dj=0.1, om0=6, normmean=True, mirrormethod=1):
	"""
		Run a double wavelet analysis on two time-series signals.

		This function applies `run_full_wavelet_analysis` to two signals and then
		computes their **cross-wavelet transform and coherence**.

		The workflow is:

		1. Perform a full wavelet analysis on each signal independently.
		2. Extract the wavelet coefficient matrices from both signals.
		3. Compute the **cross-wavelet transform**:
		Wxy = Wx * conj(Wy)
		4. Derive:

		* Cross-wavelet power
		* Rectified cross-power
		* Phase differences between the two signals
		5. Compute **wavelet coherence** using the pycwt implementation.

		This allows identification of:

		* Shared periodic signals
		* Phase relationships
		* Time-dependent coherence between the two datasets.

		## Parameters

		- signal1 : array-like. First input time series.

		- signal2 : array-like. Second input time series. Must be the same length as `signal1`.

		- dt : float, default=10000. Sampling interval of both signals.

		- mirror : bool, default=True. If True, both signals are mirrored before wavelet analysis.

		- period_min : float, optional. Minimum period for band-pass ICWT reconstruction.

		- period_max : float, optional. Maximum period for band-pass ICWT reconstruction.

		- wf : str, default='morlet'. Mother wavelet type.

		- dj : float, default=0.1. Scale resolution.

		- om0 : float, default=6. Central frequency of the wavelet.

		- normmean : bool, default=True. Whether to subtract the mean before analysis.

		- mirrormethod : int, default=1. Signal mirroring method.

		## Returns

		xmirror_df1, fft_df1, scales_df1, icwt_df1, icwt_part1_df1, wavelet_power_df1 :
		Results from the full wavelet analysis of the first signal.

		xmirror_df2, fft_df2, scales_df2, icwt_df2, icwt_part1_df2, wavelet_power_df2 :
		Results from the full wavelet analysis of the second signal.

		xypower_df : pandas.DataFrame
		Crosswavelet power spectrum.

		phasexy_df : pandas.DataFrame
		Phase differences between the two signals.

		R2ns_df : pandas.DataFrame
		Wavelet coherence values.
	"""
	len1 = len(signal1)
	len2 = len(signal2)
	if len1 != len2:
		raise ValueError('The two input signals need to be the same length!')
	if len1 == 0. or len2 == 0.:
		raise ValueError('The signals cannot have zero length.')

	# run wavelet analysis on both signals 
	result1 = run_full_wavelet_analysis(signal1, dt=dt, mirror=mirror, period_min=period_min, period_max=period_max, wf=wf, dj=dj, om0=om0, normmean=normmean, mirrormethod=mirrormethod)
	result2 = run_full_wavelet_analysis(signal2, dt=dt, mirror=mirror, period_min=period_min, period_max=period_max, wf=wf, dj=dj, om0=om0, normmean=normmean, mirrormethod=mirrormethod)
	
	(xmirror_df1, fft_df1, scales_df1, icwt_df1, icwt_part1_df1, wavelet_power_df1, coi1, cwtX, scales, x_pad) = result1
	(xmirror_df2, fft_df2, scales_df2, icwt_df2, icwt_part1_df2, wavelet_power_df2, coi2, cwtY, scales, y_pad) = result2
	
	# retriving or calcualting the relevant variables for both signals for the cross wavelet and coherence calculations  
	X = cwtX[0]
	Y = cwtY[0]

	powerx = (np.abs(X))**2.
	powery = (np.abs(Y))**2.

	## cross wavelets and coherence
	Wxy = X * np.conjugate(Y) # cross wavelet transform
	xypower = np.abs(Wxy) # cross wavelet power
	Wyy = Y * np.conjugate(Y)
	yypower = np.abs(Wyy)
	phasexy = np.angle(Wxy,deg=True) # cross wavelet phase
 

	# crosspower rectification
	xypower_rectified = xypower / scales[:, np.newaxis] # rectified cross wavelet power

	if wf == 'morlet':
		fullwavelet = pycwt.wavelet.Morlet(om0)
	elif wf == 'dog':
		fullwavelet = pycwt.wavelet.DOG(om0)
	elif wf == 'paul':
		fullwavelet = pycwt.wavelet.Paul(om0)
	else:
		sys.exit('Could not recognise desired mother wavelet, exiting...')

	print("\nCalculating coherence with pycwt...\n")
 
	R2ns, R_phase, coi_cross, freq, sig = pycwt.wct(x_pad,y_pad,dt,dj=dj,s0=scales[0],J=len(scales)-1,sig=False, wavelet=fullwavelet) # R2ns is the coherence, R_phase is the phase of the coherence, coi is the cone of influence, freq is the frequency, sig is the significance
	
 	# Save results as dataframes
	time = np.arange(X.shape[1]) * dt
	n_scales = len(scales)
	n_time = len(time)
	period = 1 / freq
 
 	# The cross wavelet power 
	xypower_df = pd.DataFrame({
		"row": np.repeat(np.arange(n_scales), n_time),
		"time": np.tile(time, n_scales),
		"xypower": xypower.flatten(),
		"xypower_rectified": xypower_rectified.flatten(),
		"period": np.repeat(period, n_time),
		"frequency": np.repeat(freq, n_time)
	})
	# The phase of the cross wavelet
	phasexy_df = pd.DataFrame({
		"row": np.repeat(np.arange(n_scales), n_time),
		"time": np.tile(time, n_scales),
		"phasexy": phasexy.flatten(),
		"period": np.repeat(period, n_time),
		"frequency": np.repeat(freq, n_time)
	})
 
	# The coherence
	R2ns_df = pd.DataFrame({
		"row": np.repeat(np.arange(n_scales), n_time),
		"time": np.tile(time, n_scales),
		"R2ns": R2ns.flatten(),
		"period": np.repeat(period, n_time),
		"frequency": np.repeat(freq, n_time)
	})
 
	return (xmirror_df1, fft_df1, scales_df1, icwt_df1, icwt_part1_df1, wavelet_power_df1, xmirror_df2, fft_df2, scales_df2, icwt_df2, icwt_part1_df2, wavelet_power_df2, xypower_df, phasexy_df, R2ns_df, coi1, coi2, coi_cross)

# calculate time-averaged power, crosspower, or coherence
def time_averaged(df, value_col, period_col='period', time_col='time'):
	"""
	Calculates time-averaged power, crosspower, or coherence for each period.

	Parameters
	----------
	df : pd.DataFrame, DataFrame containing the wavelet analysis results with columns for time, period, and the value to average.
	value_col : str, Name of the column in df that contains the values to be averaged (e.g., 'power', 'xypower', 'R2ns').
	period_col : str, Name of the column in df that contains the period values (default is 'period').
	time_col : str, Name of the column in df that contains the time values (default is 'time').

	Returns
	-------
	averaged_df : pd.DataFrame, DataFrame containing the time-averaged values for each period.
	"""
	averaged_df = df.groupby(period_col)[value_col].mean().reset_index()
	averaged_df['frequency'] = 1 / averaged_df[period_col]
	
	return averaged_df


######## unmirror defs

def unmirror_wavelet_power(sig1, wavelet_power_df_mirrored, period_min, period_max):
    """
    Removes mirroring bits from wavelet analysis results, keeping only 5th repetition.
    
    Parameters
    ----------
    sig1 : array, original (non-mirrored) signal, imput to wavelet analysis.
    wavelet_power_df_mirrored : pd.DataFrame, Mirrored wavelet power DataFrame with 'time' and 'period' columns.
    xmirror_df: pd.DataFrame, Mirror signal returned by run_full_wavelet_analysis.
    period_min : float, Minimum period to retain.
    period_max : float, Maximum period to retain.

    Returns
    -------
    wavelet_power_df : pd.DataFrame, Unmirrored, filtered wavelet power DataFrame.
    sumpower_df : pd.DataFrame, Summed/averaged power per period for the unmirrored window.
    
    """
    original_length = len(sig1)

    # Index range corresponding to the 5th repetition (central window)
    start_index = 4 * original_length
    end_index = 5 * original_length

    # --- Unmirror wavelet power ---
    extended_time = wavelet_power_df_mirrored["time"].unique()
    time_min = extended_time[start_index]
    time_max = extended_time[end_index-1]


    wavelet_power_df = wavelet_power_df_mirrored.copy()
    wavelet_power_df = wavelet_power_df[
        (wavelet_power_df["time"] >= time_min) &
        (wavelet_power_df["time"] <= time_max) &
        (wavelet_power_df["period"] >= period_min) &
        (wavelet_power_df["period"] <= period_max)
    ]

    # Restore time axis to start from 0
    wavelet_power_df["time"] = wavelet_power_df["time"] - time_min
    wavelet_power_df = wavelet_power_df.reset_index(drop=True)

    # Recompute summed power over the unmirrored window
    sumpower_df = wavelet_power_df.groupby("period")[["rectified_power", "power"]].mean().reset_index()
    sumpower_df["frequency"] = 1 / sumpower_df["period"]

    return wavelet_power_df, sumpower_df

def unmirror_icwt(sig1, icwt_df, icwt_band1_df, xmirror_df):
    """
    Removes mirroring bits from ICWT results, keeping only the 5th repetition.

    Parameters
    ----------
    sig1 : array, Original (non-mirrored) signal, input to wavelet analysis.
    icwt_df : pd.DataFrame, Inverse CWT of the full mirrored signal.
    icwt_band1_df : pd.DataFrame, Inverse CWT of the band-filtered mirrored signal.
    xmirror_df : pd.DataFrame, Mirror signal returned by run_full_wavelet_analysis.

    Returns
    -------
    icwt_unmirrored : pd.DataFrame, Reconstructed signal from the 5th repetition, restored to original scale.
    icwt_band1_unmirrored : pd.DataFrame, Band-filtered reconstructed signal, restored to original scale.
    """
    original_length = len(sig1)
    sig1_mean = np.mean(sig1)
    mirror_offset = xmirror_df.iloc[0, 0]

    # Index range corresponding to the 5th repetition (central window)
    start_index = 4 * original_length
    end_index = 5 * original_length

    # --- Unmirror icwt ---
    icwt_unmirrored = icwt_df + mirror_offset
    icwt_unmirrored = icwt_unmirrored.iloc[start_index:end_index]
    icwt_unmirrored = icwt_unmirrored + sig1_mean

    # --- Unmirror icwt_band1 ---
    icwt_band1_unmirrored = icwt_band1_df.iloc[start_index:end_index]
    icwt_band1_unmirrored = icwt_band1_unmirrored + sig1_mean + mirror_offset

    return icwt_unmirrored, icwt_band1_unmirrored

def unmirror_crosspower(sig1, xypower_df, period_min, period_max):
    """
    Removes mirroring bits from wavelet analysis results, keeping only 5th repetition.
    
    Parameters
    ----------
    sig1 : array, original (non-mirrored) signal, imput to wavelet analysis.
    xypower_df : pd.DataFrame, Cross-wavelet power DataFrame with 'time', 'period', and 'power' columns.
    period_min : float, Minimum period to retain.
    period_max : float, Maximum period to retain.

    Returns
    -------
    crosspower_df : pd.DataFrame, Unmirrored, filtered cross-wavelet power DataFrame.
    time_averaged_crosspower : pd.DataFrame, Time-averaged cross-wavelet power per period for the unmirrored window.
    
    """
    original_length = len(sig1)

    # Index range corresponding to the 5th repetition (central window)
    start_index = 4 * original_length
    end_index = 5 * original_length

    # --- Unmirror wavelet power ---
    extended_time = xypower_df["time"].unique()
    time_min = extended_time[start_index]
    time_max = extended_time[end_index-1]


    crosspower_df = xypower_df.copy()
    crosspower_df = crosspower_df[
        (crosspower_df["time"] >= time_min) &
        (crosspower_df["time"] <= time_max) &
        (crosspower_df["period"] >= period_min) &
        (crosspower_df["period"] <= period_max)
    ]

    # Restore time axis to start from 0
    crosspower_df["time"] = crosspower_df["time"] - time_min
    crosspower_df = crosspower_df.reset_index(drop=True)

    # Recompute summed power over the unmirrored window
    time_averaged_crosspower = crosspower_df.groupby("period")[["xypower_rectified", "xypower"]].mean().reset_index()
    time_averaged_crosspower["frequency"] = 1 / time_averaged_crosspower["period"]

    return crosspower_df, time_averaged_crosspower

def unmirror_coherence(sig1, R2ns_df, period_min, period_max):
	"""
	Removes mirroring bits from coherence results, keeping only 5th repetition.
	
	Parameters
	----------
	sig1 : array, original (non-mirrored) signal, imput to wavelet analysis.
	R2ns_df : pd.DataFrame, Coherence DataFrame with 'time', 'period', and 'R2ns' columns.
	period_min : float, Minimum period to retain.
	period_max : float, Maximum period to retain.

	Returns
	-------
	coherence_df : pd.DataFrame, Unmirrored, filtered coherence DataFrame.
	time_averaged_coherence : pd.DataFrame, Time-averaged coherence per period for the unmirrored window.
	
	"""
	original_length = len(sig1)

	# Index range corresponding to the 5th repetition (central window)
	start_index = 4 * original_length
	end_index = 5 * original_length

	# --- Unmirror coherence ---
	extended_time = R2ns_df["time"].unique()
	time_min = extended_time[start_index]
	time_max = extended_time[end_index-1]


	coherence_df = R2ns_df.copy()
	coherence_df = coherence_df[
		(coherence_df["time"] >= time_min) &
		(coherence_df["time"] <= time_max) &
		(coherence_df["period"] >= period_min) &
		(coherence_df["period"] <= period_max)
	]

	# Restore time axis to start from 0
	coherence_df["time"] = coherence_df["time"] - time_min
	coherence_df = coherence_df.reset_index(drop=True)

	# Recompute averaged coherence over the unmirrored window
	time_averaged_coherence = coherence_df.groupby("period")[["R2ns"]].mean().reset_index()
	time_averaged_coherence["frequency"] = 1 / time_averaged_coherence["period"]

	return coherence_df, time_averaged_coherence

def unmirror_phase(sig1, phasexy_df, period_min, period_max):
	"""
	Removes mirroring bits from phase results, keeping only 5th repetition.
	
	Parameters
	----------
	sig1 : array, original (non-mirrored) signal, imput to wavelet analysis.
	phasexy_df : pd.DataFrame, Phase DataFrame with 'time', 'period', and 'phasexy' columns.
	period_min : float, Minimum period to retain.
	period_max : float, Maximum period to retain.

	Returns
	-------
	phase_df : pd.DataFrame, Unmirrored, filtered phase DataFrame.
	time_averaged_phase : pd.DataFrame, Time-averaged phase per period for the unmirrored window.
	
	"""
	original_length = len(sig1)

	# Index range corresponding to the 5th repetition (central window)
	start_index = 4 * original_length
	end_index = 5 * original_length

	# --- Unmirror phase ---
	extended_time = phasexy_df["time"].unique()
	time_min = extended_time[start_index]
	time_max = extended_time[end_index-1]


	phase_df = phasexy_df.copy()
	phase_df = phase_df[
		(phase_df["time"] >= time_min) &
		(phase_df["time"] <= time_max) &
		(phase_df["period"] >= period_min) &
		(phase_df["period"] <= period_max)
	]

	# Restore time axis to start from 0
	phase_df["time"] = phase_df["time"] - time_min
	phase_df = phase_df.reset_index(drop=True)
 
	return phase_df