#!/usr/bin/env python
# coding: utf-8

import sys
import time
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
from museval.metrics import bss_eval_images, bss_eval_sources

### Function for audio pre-processing ###
def pre_processing(data, Fs, down_sam):
    
    #Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    
    #Down sampling and normalization of the wave
    if down_sam is not None:
        wavdata = sg.resample_poly(wavdata, down_sam, Fs)
        Fs = down_sam
    
    return wavdata, Fs

### Function for getting STFT ###
def get_STFT(wav, Fs, frame_length, frame_shift):
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Execute STFT
    freqs, times, Y = sg.stft(wav, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    
    #Display the size of input
    print("Spectrogram size (freq, time) = " + str(Y.shape))
    
    return Y, Fs, freqs, times

### Function for getting inverse STFT ###
def get_invSTFT(Y, Fs, frame_length, frame_shift):
    
    #Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    
    return rec_wav, Fs

### Function for removing components closing to zero ###
def get_nonzero(tensor):
    tensor = np.where(np.abs(tensor) < 1e-10, tensor+1e-10, tensor)
    return tensor

### Function for reconstructing wave from H, U, P ###
def get_rec(H, U, P):
    
    #Reconstruct wave according to the signal model
    Y = H[:, :, np.newaxis] * U[np.newaxis, :, :] * P
    Y = np.sum(Y, axis=1)
    return Y

### Function for getting basements and weights matrix by NMF ###
def get_complexNMF(Y, num_iter, num_base, loss_func, d):
    
    #Caution! In this code (complex NMF), the valuable shoud be tensor
    #Y, H, U are 2D-matrix, P, beta, X are 3D-tensor
    
    #Initialize basements and weights based on the Y size(k, n)
    K, N = Y.shape[0], Y.shape[1]
    if num_base >= K or num_base >= N:
        print("The number of basements should be lower than input size.")
        sys.exit()
    
    #Initialize basements H and activation U
    H = np.random.rand(K, num_base) #basements (distionaries)
    U = np.random.rand(num_base, N) #weights (coupling coefficients)
    
    #Initialize loss
    loss = np.zeros(num_iter)
    
    #For a progress bar
    unit = int(np.floor(num_iter/10))
    bar = "#" + " " * int(np.floor(num_iter/unit))
    start = time.time()
    
    #In the case of squared Euclidean distance
    if loss_func == "EU":
        
        #Set sparse parameter lambda to be sum(|Y|^2) * 1e-5 / (num_base)^(1-order/2)
        lamb = np.sum(np.abs(Y)**2) * 1e-5 / num_base**(1-d/2)
        
        #Initialize phase P as Y/|Y|
        P = (Y / np.abs(Y))[:, np.newaxis, :]
        
        #Repeat num_iter times
        for i in range(num_iter):
            
            #Display a progress bar
            print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i, num_iter), end="")
            if i % unit == 0:
                bar = "#" * int(np.ceil(i/unit)) + " " * int(np.floor((num_iter-i)/unit))
                print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i, num_iter), end="")
            
            #Update beta tensor
            beta = (H[:, :, np.newaxis] * U[np.newaxis, :, :]) / get_nonzero((H @ U)[:, np.newaxis, :])
            beta = get_nonzero(beta) #Avoid zero divide
            
            #Update X tensor
            F = H[:, :, np.newaxis] * U[np.newaxis, :, :] * P
            X = F + beta * (Y[:, np.newaxis, :] - np.sum(F, axis=1, keepdims=True))
            
            #Update phase P tensor
            P = get_nonzero(X)
            P = P / np.abs(P)
            
            #Update the basements
            numer = np.sum((U[np.newaxis, :, :] * np.abs(X)) / beta, axis=2)
            denom = np.sum((U[np.newaxis, :, :])**2 / beta, axis=2)
            H = numer / get_nonzero(denom)
            #Normalization
            H = H / np.sum(H, axis=0, keepdims=True)
            
            #Update the weights
            numer = np.sum((H[:, :, np.newaxis] * np.abs(X)) / beta, axis=0)
            denom = np.sum((H[:, :, np.newaxis])**2 / beta, axis=0)
            denom = denom + lamb*d*np.abs(U)**(d-2)
            U = numer / get_nonzero(denom)
            
            #Compute the loss function
            loss[i] = np.sum(np.abs(Y - get_rec(H, U, P))**2) + 2*lamb*np.sum(np.abs(U)**d)
    
    #In the case of Kullback–Leibler divergence
    elif loss_func == "KL":
        
        #Set a regularization factor lambda (for sparse term)
        lamb = 1
        
        #Initialize X for dual complex-NMF problem
        beta = (H[:, :, np.newaxis] * U[np.newaxis, :, :]) / get_nonzero((H @ U)[:, np.newaxis, :])
        F = H[:, :, np.newaxis] * U[np.newaxis, :, :] * (Y / np.abs(Y))[:, np.newaxis, :]
        X = F + beta * (Y[:, np.newaxis, :] - np.sum(F, axis=1, keepdims=True))
        Xabs = get_nonzero(np.abs(X))
        
        #Repeat num_iter times
        for i in range(num_iter):
            
            #Display a progress bar
            print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i, num_iter), end="")
            if i % unit == 0:
                bar = "#" * int(np.ceil(i/unit)) + " " * int(np.floor((num_iter-i)/unit))
                print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i, num_iter), end="")
            
            #Get absolute amplitude of the X
            Xabs = get_nonzero(np.abs(X))
            
            #Update D tensor
            HU = get_nonzero(H[:, :, np.newaxis] * U[np.newaxis, :, :])
            D = np.log(Xabs / HU) - 2
            
            #Update A tensor (this parameter needs to distinguish as for D)
            A = np.where(D<0, 1/Xabs, 0.5*D/Xabs+1/Xabs)
            
            #Update B tensor (this parameter needs to distinguish as for D)
            B = np.where(D<0, -0.5*D*X/Xabs, 0)
            
            #Update X tensor and phase P
            X = (B + ((Y - (B/A).sum(axis=1)) / (1/A).sum(axis=1))[:, np.newaxis, :]) / A
            Xabs = get_nonzero(np.abs(X))
            P = X / Xabs
            
            #Update the basements H
            numer = np.sum(Xabs, axis=2)
            denom = np.sum(U, axis=1)[np.newaxis, :]
            H = numer / get_nonzero(denom)
            #Normalization
            H = H / np.sum(H, axis=0, keepdims=True)
            
            #Update the weights U
            numer = np.sum(Xabs, axis=0)
            denom = np.sum(H, axis=0)[:, np.newaxis] + lamb*d*np.abs(get_nonzero(U))**(d-1)
            U = numer / get_nonzero(denom)
            
            #Compute the loss function
            HU = get_nonzero(H[:, :, np.newaxis] * U[np.newaxis, :, :])
            loss[i] = np.sum(Xabs*np.log(Xabs / HU) - Xabs + HU) + 2*lamb*np.sum(np.abs(U)**d)
    
    #Finish the progress bar
    bar = "#" * int(np.ceil(num_iter/unit))
    print("\rProgress:[{0}] {1}/{2} {3:.2f}sec Completed!".format(bar, i+1, num_iter, time.time()-start), end="")
    print()
    
    return H, U, P, loss

### Function for decompose spectrogram into each basement ###
def get_decompose(H, U, P, num_base, Fs, frame_length, frame_shift):
    
    #Pick up each dictionary
    for j in range(num_base):
        zero_mat = np.zeros((H.shape[1], H.shape[1]))
        zero_mat[j, j] = 1
        Base = get_rec(H @ zero_mat, U, P) #Extract an only dictionary
        Base_wav, Fs = get_invSTFT(Base, Fs, frame_length, frame_shift)
        print("Basement " + str(j+1))
        sf.write("./result/Basement_{}.wav".format(j+1), Base_wav, Fs)
        if j ==0:
            Sum_wav = Base_wav
        else:
            Sum_wav = Sum_wav + Base_wav
    
    return

### Function for plotting Spectrogram and loss curve ###
def display_graph(Y, X, times, freqs, loss_func, num_iter):
    
    #Plot the original spectrogram
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.title('An original spectrogram')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.pcolormesh(times, freqs, 10*np.log10(np.abs(Y)), cmap='jet')
    plt.colorbar(orientation='horizontal').set_label('Power')
    plt.savefig("./result/original_spec.png", dpi=200)
    
    #Plot the reconstructed spectrogram
    plt.subplot(1, 2, 2)
    plt.title('The approximation by complex NMF')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.pcolormesh(times, freqs, 10*np.log10(np.abs(X)), cmap='jet')
    plt.colorbar(orientation='horizontal').set_label('Power')
    plt.savefig("./result/reconstructed_spec.png", dpi=200)
    
    #Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, num_iter+1), loss[:], marker='.')
    plt.title(loss_func + '_loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.savefig("./result/loss_curve.png", dpi=200)
    
    return

### Main ###
if __name__ == "__main__":
    
    #Setup
    down_sam = None        #Downsampling rate (Hz) [Default]None
    frame_length = 0.064   #STFT window width (second) [Default]0.064
    frame_shift = 0.032    #STFT window shift (second) [Default]0.032
    num_iter = 100         #The number of iteration [Default]100
    num_base = 3           #The number of basements [Default]3
    spa_order = 1.0        #The order (<=1.0) of activation sparsity (2λ|U|^d) [Default]1.0
    loss_func = "EU"       #Select either EU or KL divergence [Default]EU
    
    #Define random seed
    np.random.seed(seed=32)
    
    #Read a sound file
    source = "./data/piano.wav"
    data, Fs = sf.read(source)
    
    #Call my function for audio pre-processing
    data, Fs = pre_processing(data, Fs, down_sam)
    
    #Call my function for getting STFT (complex STFT amplitude)
    print("Original sound")
    sf.write("./result/Original_sound.wav", data, Fs)
    Y, Fs, freqs, times = get_STFT(data, Fs, frame_length, frame_shift)
    
    #Call my function for updating NMF basements and weights
    H, U, P, loss = get_complexNMF(Y, num_iter, num_base, loss_func, spa_order)
    
    #Call my function for getting inverse STFT
    X = get_rec(H, U, P)
    rec_wav, Fs = get_invSTFT(X, Fs, frame_length, frame_shift)
    rec_wav = rec_wav[: int(data.shape[0])] #inverse stft includes residual part due to zero padding
    print("Reconstructed sound")
    sf.write("./result/Reconstructed_sound.wav", rec_wav, Fs)
    
    #Call my function for decomposing basements
    get_decompose(H, U, P, num_base, Fs, frame_length, frame_shift)
    
    #Compute the SDR by bss_eval from museval library ver.4
    data = data[np.newaxis, :, np.newaxis]
    rec_wav = rec_wav[np.newaxis, :, np.newaxis]
    sdr, isr, sir, sar, perm = bss_eval_images(data, rec_wav)
    #sdr, sir, sar, perm = bss_eval_sources(truth, data) #Not recommended by documentation
    print("SDR: {:.3f} [dB]".format(sdr[0, 0]))
    
    #Call my function for displaying graph
    display_graph(Y, X, times, freqs, loss_func, num_iter)