# Pill Classification

## DONE

### 1. Data Acquiration

      - link: http://drug.mfds.go.kr/html/index.jsp#

### 2. Data preprocess (except Pill Mask)

      - Image Crop/Resizing ( Final Image Size: 1024 × 512 × 3 )

      - Label naming

### 3. Deep Neural Network Train Frame-work (with Pytorch)

      - Model: Baisc ResNet18
      
      - Augmentation: Vertical, Horizontal Flip / Tilt / Scaling / Shear

### 4. Shape(11types) Train 

      - Validation Acc: 87% (2 epoch train)

### 5. Front Color(16types) Train 

      - Validation Acc: 78% (2 epoch train)

### 6. Back Color(16types) Train 

      - Validation Acc: 
<br>

## TODO

### 1. Test.py

### 2. Multi-Instance, Multi-Label Learning

### 3. Pill Mask 
      - Reference Paper: 
      Real-world Pill Segmentation based on Superpixel Merge using Region Adjacency Graph
      link: http://www.scitepress.org/Papers/2017/61358/61358.pdf
