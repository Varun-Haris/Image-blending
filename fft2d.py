import numpy as np
import scipy.stats as st

# Gaussian Kernel generation 
def gkern(kernlen=21, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()

class fourier():
    def dft(self, img):
        trans = []
        dft = []
        for i in range(0,img.shape[0]):
            trans.append(np.fft.fft(img[i,:]))
    
        for j in range(0,img.shape[1]):
            dft.append(np.fft.fft(np.array(trans)[:,j]))
        
        return np.array(dft).T

    def idft(self, img):
        conjugate = np.conj(img)
        print(conjugate.dtype)
        reconstruct = self.dft(conjugate)/(img.shape[0]*img.shape[1])
        print (reconstruct.dtype)
                
        return reconstruct
