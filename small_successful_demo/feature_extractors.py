import torch
import numpy as np
import scipy.signal
import pywt

#power spectral density feature extractor for eeg frequency band analysis  
class PSDExtractor:
    def __init__(self, fs=250):
        self.fs = fs
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    #extract power spectral density features from eeg windows - vectorized for batch processing
    def extract_features(self, eeg_tensor):
        #eeg_tensor shape: (batch, window_size, channels)
        batch_size, window_size, channels = eeg_tensor.shape
        
        #reshape to process all batch*channel combinations at once
        eeg_reshaped = eeg_tensor.permute(0, 2, 1).reshape(batch_size * channels, window_size)
        eeg_numpy = eeg_reshaped.cpu().numpy()
        
        #vectorized psd computation across all channels and batches
        freqs, psd = scipy.signal.welch(
            eeg_numpy,
            fs=self.fs,
            nperseg=window_size,
            noverlap=0,
            axis=1  #compute along time dimension
        )
        
        #extract band powers vectorized
        band_powers = []
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                #integrate power in frequency band for all signals simultaneously
                band_power = np.trapz(psd[:, band_mask], freqs[band_mask], axis=1)
            else:
                band_power = np.zeros(batch_size * channels)
            band_powers.append(band_power)
        
        #reshape back to (batch, channels, frequency_bands)
        band_powers = np.array(band_powers).T  #shape: (batch*channels, 5)
        band_powers = band_powers.reshape(batch_size, channels, 5)
        
        return torch.FloatTensor(band_powers)

#fast fourier transform feature extractor for detailed frequency spectrum analysis
class FFTExtractor:
    def __init__(self, fs=250, max_freq=50):
        self.fs = fs  #sampling frequency
        self.max_freq = max_freq  #maximum frequency to include
    
    #extract fft magnitude spectrum features from eeg windows
    def extract_features(self, eeg_tensor):
        #eeg_tensor shape: (batch, window_size, channels)
        batch_size, window_size, channels = eeg_tensor.shape
        
        #compute real fft along time dimension
        fft_result = torch.fft.rfft(eeg_tensor, dim=1)
        
        #get magnitude spectrum
        magnitude_spectrum = torch.abs(fft_result)
        
        #create frequency bins
        freqs = torch.fft.rfftfreq(window_size, d=1/self.fs)
        
        #limit to max frequency
        freq_mask = freqs <= self.max_freq
        magnitude_spectrum = magnitude_spectrum[:, freq_mask, :]
        
        #transpose to match output format: (batch, channels, frequency_bins)
        magnitude_spectrum = magnitude_spectrum.transpose(1, 2)
        
        return magnitude_spectrum

#discrete wavelet transform feature extractor for multi-resolution time-frequency analysis
class DWTExtractor:
    def __init__(self, wavelet='db4', levels=3):
        self.wavelet = wavelet
        self.levels = levels
    
    #extract dwt coefficients from eeg windows - vectorized for batch processing
    def extract_features(self, eeg_tensor):
        #eeg_tensor shape: (batch, window_size, channels)
        batch_size, window_size, channels = eeg_tensor.shape
        
        #reshape to process all batch*channel combinations at once
        eeg_reshaped = eeg_tensor.permute(0, 2, 1).reshape(batch_size * channels, window_size)
        eeg_numpy = eeg_reshaped.cpu().numpy()
        
        #vectorized dwt computation - process all signals simultaneously
        all_coeffs = []
        for i in range(eeg_numpy.shape[0]):
            coeffs = pywt.wavedec(
                eeg_numpy[i, :], 
                wavelet=self.wavelet, 
                level=self.levels
            )
            
            #extract statistical features from each coefficient set
            coeff_features = []
            for coeff_array in coeffs:
                coeff_features.append(np.mean(np.abs(coeff_array)))
            
            all_coeffs.append(coeff_features)
        
        #convert to numpy array and reshape back to (batch, channels, coefficients)
        all_coeffs = np.array(all_coeffs)  #shape: (batch*channels, levels+1)
        all_coeffs = all_coeffs.reshape(batch_size, channels, self.levels + 1)
        
        return torch.FloatTensor(all_coeffs)

#unified eeg feature extractor supporting multiple extraction methods
class EEGFeatureExtractor:
    def __init__(self, method='psd', **kwargs):
        self.method = method
        
        if method == 'psd':
            self.extractor = PSDExtractor(**kwargs)
        elif method == 'fft':
            self.extractor = FFTExtractor(**kwargs)
        elif method == 'dwt':
            self.extractor = DWTExtractor(**kwargs)
        else:
            raise ValueError(f"unsupported extraction method: {method}")
    
    #extract features using selected method
    def __call__(self, eeg_tensor):
        return self.extractor.extract_features(eeg_tensor)
    
    #get feature dimension for model configuration
    def get_feature_dim(self, window_size=200):
        if self.method == 'psd':
            return 5  #five frequency bands
        elif self.method == 'fft':
            #frequency bins up to max_freq
            max_freq = getattr(self.extractor, 'max_freq', 50)
            fs = getattr(self.extractor, 'fs', 250)
            freq_bins = int((max_freq / (fs/2)) * (window_size//2 + 1))
            return freq_bins
        elif self.method == 'dwt':
            return self.extractor.levels + 1  #approximation + detail coefficients
        
        return 0