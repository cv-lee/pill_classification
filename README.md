# Pill Classification

## DONE

1. Data Acquiration

      - link: http://drug.mfds.go.kr/html/index.jsp#

<br>

2. Data preprocess (except Pill Mask)

      - Image Crop/Resizing ( Final Image Size: 1024 × 512 × 3 )

      - Label naming

<br>

3. Deep Neural Network Train Frame-work (with Pytorch)

      - Model: Baisc ResNet18
      
      - Augmentation: Vertical, Horizontal Flip / Tilt / Scaling / Shear
<br>

4. Shape(11types) Train 

      - Validation Acc: 87% (2 epoch train)
<br>

5. Front Color(16types) Train 

      - Validation Acc: 78% (2 epoch train)
<br>

6. Back Color(16types) Train 

      - Validation Acc: 
<br>
<br>

## TODO

1. Testset / Test.py

2. Multi-Instance, Multi-Label Learning

3. Pill Mask 
