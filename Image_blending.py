import cv2
import numpy as np
import matplotlib.pyplot as plt
import Gauss_laplace_pyramids

#def isValid(m,n,)

def onMouse(event,x,y,flags,userdata):
    global contours, pts, cropping, image, done
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
        
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        contours.append(pts)
            
    if cropping:
        pts.append((x,y))
        
def alignImages(fg, bg, pair):
    ## Wolfpack on body_builder
    if pair == 1:
        fg = np.pad(fg, ((0,275),(157,118)), 'constant')
    
    ## Emma on wolfpack
    if pair == 2:
        fg = np.pad(fg, ((0,125),(55,70)), 'constant')
    
    ## varun on body_builder
    if pair == 3:
        fg = np.pad(fg, ((0,260),(150,110)), 'constant')
        
    return fg, bg    
        
def blendImage(bg, fg, region, layers):
    gpyr_bg, lpyr_bg = Gauss_laplace_pyramids.computePyr(bg, layers)
    gpyr_fg, lpyr_fg = Gauss_laplace_pyramids.computePyr(fg, layers)
    
    gpyr = list(Gauss_laplace_pyramids.computeGaussPyr(region, layers))
    k = len(gpyr)
    
    lpyr_bg.append(gpyr[-1])
    lpyr_fg.append(gpyr[-1])
    
    for i in range(k):
        if np.shape(gpyr[i]) != np.shape(lpyr_bg[i]) or np.shape(gpyr[i]) != np.shape(lpyr_bg[i]):
            gpyr[i] = np.delete(gpyr[i],(-1),axis=0)
            gpyr[i] = np.delete(gpyr[i],(-1),axis=1)
            
        LS = gpyr[i]*lpyr_fg[i] + (1-gpyr[i])*lpyr_bg[i]
        yield LS
        
        
def collapse(blended):
    output = np.zeros((blended[0].shape[0],blended[0].shape[1]), dtype=np.float64)
    blended = list(blended)
    
    for i in range(len(blended)-1,0,-1):
        lap = Gauss_laplace_pyramids.upSample(np.array(blended[i]))
        lapb = np.array(blended[i-1])
        if lap.shape[0] < lapb.shape[0]:
            lap = np.pad(lap,((1,1),(1,1)),'reflect')
            
        tmp = lap + lapb
        
        blended.pop()
        blended.pop()
        blended.append(tmp)
        output = tmp
            
    return output

########
### DRIVER CODE
########
    
layers = int(input("Enter the number of layers for pyramid computation: "))
pair = int(input("Enter image pair (1, 2, or 3): "))
if pair == 1:
    bg_img = cv2.imread(r'E:\NCSU\Fall-19\Digital Imaging systems\Projects\Project3\body_builder.png')
    fg_image = cv2.imread(r'E:\NCSU\Fall-19\Digital Imaging systems\Projects\Project3\wolfpack.png')
    fg_gray = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    
elif pair == 2:
    bg_img = cv2.imread(r'E:\NCSU\Fall-19\Digital Imaging systems\Projects\Project3\wolfpack.png')
    fg_image = cv2.imread(r'E:\NCSU\Fall-19\Digital Imaging systems\Projects\Project3\emma.png')
    fg_gray = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    
elif pair == 3:
    bg_img = cv2.imread(r'E:\NCSU\Fall-19\Digital Imaging systems\Projects\Project3\body_builder.png')
    fg_image = cv2.imread(r'E:\NCSU\Fall-19\Digital Imaging systems\Projects\Project3\varun.png')
    fg_gray = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)

levels = Gauss_laplace_pyramids.isValid(bg_gray.shape[0], bg_gray.shape[1], layers)
result = bg_img.copy()
contours = []
pts = []
cropping = False
done = False

### Image alignment for blending
fg_gray, bg_gray = alignImages(fg_gray, bg_gray, pair)
rf,gf,bf = cv2.split(fg_image)
rb,gb,bb = cv2.split(bg_img)
rf,rb = alignImages(rf,rb, pair)
gf,gb = alignImages(gf,gb, pair)
bf,bb = alignImages(bf,bb, pair)

### Contour extraction for mask
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", onMouse)

while True:
    cv2.imshow("Image", fg_gray)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("c"):
        print("Done !!")
        break
        
cv2.waitKey(1)
cv2.destroyAllWindows()

### Image padding and alignment with background
image = bg_gray.copy()

### Mask generation from the extracted contour
print("Generating Mask. \nThis may take a while.")
mask = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8())
cv2.drawContours(mask, np.array(contours), -1, (255), thickness=cv2.FILLED)
            
plt.imshow(mask, cmap='gray')
plt.title("Generated mask")
plt.show()

### Final image blending and reconstruction
mode = input('Color or Grayscale (C/G): ')
print("Blending the two !!")

if mode == 'C':
    rf = rf.astype(float)
    gf = gf.astype(float)
    bf = bf.astype(float)
 
    rb = rb.astype(float)
    gb = gb.astype(float)
    bb = bb.astype(float)
    
    mask = mask.astype(float)/255
    
    blended = list(blendImage(rb, rf, mask, levels))
    image_final_1 = np.array(collapse(np.array(blended)))
    image_final_1[image_final_1 < 0] = 0
    image_final_1[image_final_1 > 255] = 255
    image_final_1 = image_final_1.astype(np.uint8)
    
    blended = list(blendImage(gb, gf, mask, levels))
    image_final_2 = np.array(collapse(np.array(blended)))
    image_final_2[image_final_2 < 0] = 0
    image_final_2[image_final_2 > 255] = 255
    image_final_2 = image_final_2.astype(np.uint8)
    
    
    blended = list(blendImage(bb, bf, mask, levels))
    image_final_3 = np.array(collapse(np.array(blended)))
    image_final_3[image_final_3 < 0] = 0
    image_final_3[image_final_3 > 255] = 255
    image_final_3 = image_final_3.astype(np.uint8)

    
    result = np.zeros(bg_img.shape,dtype=bg_img.dtype)
    tmp = []
    tmp.append(image_final_1)
    tmp.append(image_final_2)
    tmp.append(image_final_3)
    result = cv2.merge(tmp,result)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
elif mode == 'G':
    mask = mask.astype(float)/255
    blended = list(blendImage(bg_gray, fg_gray, mask, levels))
    image_final = np.array(collapse(np.array(blended)))
    plt.imshow(image_final, cmap='gray')
