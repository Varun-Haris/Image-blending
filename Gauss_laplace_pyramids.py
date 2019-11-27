import numpy as np
import fft2d

trans = fft2d.fourier()
p = 21
q = 21
kernel = fft2d.gkern()

def isValid(m,n,layers):   
    for i in range(1,layers+1):
        if int(m/(2**i)) > 10:
            continue
        else:
            print("Max layers reached. Limit = {}".format(i-1))
            return i-1
    return layers
    
def downSample(img):
    global p, q, kernel, trans
    m,n = img.shape
    
    kernel = np.pad(kernel,
                    ((np.abs(int(m/2)-int(p/2)), np.abs(int(m/2)-int(p/2)-1)),
                     (np.abs(int(m/2)-int(p/2)), np.abs(int(m/2)-int(p/2)-1))),
                    'constant').astype(float)
    
    if kernel.shape[0] < img.shape[0]:
        img = np.delete(img,(-1),axis=0)
    if kernel.shape[1] < img.shape[1]:
        img = np.delete(img,(-1),axis=1)

    fftConv = np.fft.fftshift(trans.dft(kernel)*trans.dft((img/255).astype(float)))
    newImg = np.absolute(trans.idft(fftConv)).astype(float)
    kernel = fft2d.gkern()
    
    rowSlice = newImg[::2,:]
    return rowSlice[:,::2] 

def upSample(img):
    global p, q, kernel, trans
    #gauss_filter = imageConv()     
    newImage = np.zeros((img.shape[0]*2, img.shape[1]*2))
    newImage[::2,::2] = img[:,:]
    
    m,n = newImage.shape
    kernel = np.pad(kernel,
                    ((np.abs(int(m/2)-int(p/2)), np.abs(int(m/2)-int(p/2)-1)),
                     (np.abs(int(m/2)-int(p/2)), np.abs(int(m/2)-int(p/2)-1))),
                    'constant').astype(float)
    
    fftConv = np.fft.fftshift(trans.dft(kernel)*trans.dft((newImage/255).astype(float)))
    newImg = np.absolute(trans.idft(fftConv)).astype(float)
    kernel = fft2d.gkern()
    #newImage = gauss_filter.convImg(newImage, 'gauss2D', 'reflect')

    return newImg
    
def computeGaussPyr(img, levels):   
    down = img.copy()
    yield img
    for i in range(0,levels):
        down = downSample(down)
        yield down
        
def computeLaplacePyr(img, levels):
    out = []
    for i in range(len(img)-1):
        up = upSample(img[i+1])
        if up.shape[0] < img[i].shape[0]:
            img[i] = np.delete(img[i],(-1),axis=0)
        if up.shape[1] < img[i].shape[1]:
            img[i] = np.delete(img[i],(-1),axis=1)
        out = img[i] - up
        yield out
        

def computePyr(img, layers):
    m,n = img.shape
    levels = isValid(m,n,layers)
    
    gpyr = list(computeGaussPyr(img, levels))
    lpyr = list(computeLaplacePyr(gpyr, levels))
    return gpyr, lpyr
