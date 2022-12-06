import os
import cv2

image_fns = os.listdir('images')

arr = []
img = None

def mouse_click(event, x, y, 
                flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(arr) > 0:
            cv2.line(img, arr[-1], (x, y), (255, 0, 0), 3)
            cv2.imshow('image', img)
        arr.append( (x, y) )
        
for image_fn in image_fns:
    data_fn = os.path.join('data', image_fn.split('.')[0]+'.txt')
    if os.path.exists(data_fn):
        continue
    img = cv2.imread(os.path.join('images', image_fn))
    arr = []
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse_click)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

    arr = arr[:-1] # arr nin son elemanini sil

    res = ""
    for x, y in arr:
        res = res + str(x) + " " + str(y) + "\n"
    
    open(data_fn, 'w').write(res)
