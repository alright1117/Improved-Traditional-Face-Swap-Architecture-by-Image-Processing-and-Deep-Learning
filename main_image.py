import cv2
from swapface import swapface

if __name__ == '__main__' :
    

    src_path = 'src/lee.jpg'
    dst_path = 'src/IU.jpg'
    
    img1 = cv2.imread(src_path)
    img2 = cv2.imread(dst_path)

    output, _ = swapface(img1 , img2, SBR=True, hm=False, pe=True)
    cv2.imwrite("result/SBR-PE.png", output)