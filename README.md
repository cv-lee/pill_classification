# Pill Classification

## DONE

### 1. Data Acquiration

      Link: http://drug.mfds.go.kr/html/index.jsp#

### 2. Data preprocess (except Pill Mask)

      Image Crop/Resizing ( Final Image Size: 1024 × 512 × 3 )

      Label Naming

### 3. Deep Neural Network Train Frame-work (with Pytorch)

      Model: Baisc ResNet18
      
      Augmentation: Vertical, Horizontal Flip / Tilt / Scaling / Shear

### 4. Train 

#### 4-1) 1 instance → 1 label
      
      Shape (11 Types) 
      Validation Acc: 87% (2 epoch train)

      Front Color (16 Types) 
      Validation Acc: 80% (4 epoch train)

      Back Color (16 Types)
      Validation Acc: 79% (4 epoch train)
      
#### 4-2) 1 instance → 3 label (Multi-Label Learning)
      
      (5 epoch train)
      
      Shape (11 Types) 
      Validation Acc: 87% 

      Front Color (16 Types) 
      Validation Acc: 87% 

      Back Color (16 Types)
      Validation Acc: 88%

      → Total Acc: 66%   (0.87 × 0.87 × 0.88)
<br>

## TODO 

### 1. Pill Mask 
      Reference Paper: 
      
      1. Real-world Pill Segmentation based on Superpixel Merge using Region Adjacency Graph
         (Link: http://www.scitepress.org/Papers/2017/61358/61358.pdf)


### 2. Multi-Instance, Multi-Label Learning

![](https://i.imgur.com/7IVs7jC.png)

<br>

### 3. Test.py (Testset+test)
