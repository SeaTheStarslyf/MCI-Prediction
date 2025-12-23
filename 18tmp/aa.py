import cv2
import os

for i in range(9):
    img_path = '/data-pool/data/data2/qiuhui/code/Alifuse_bibm/18tmp/{}.png'.format(str(i*2+1))
    mask_path = '/data-pool/data/data2/qiuhui/code/Alifuse_bibm/18tmp/{}.png'.format(str(i*2+2))
    print(img_path)
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            if img[j,k].sum()==0:
                mask[j,k] = [98,2,0]
    cv2.imwrite(mask_path.replace('.png','_v2.png'),mask)