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

![](https://i.imgur.com/yfpsIY4.png)

<br>

      Shape (11 Types) 
      Validation Acc: 87% (2 epoch train)

      Front Color (16 Types) 
      Validation Acc: 80% (4 epoch train)

      Back Color (16 Types)
      Validation Acc: 79% (4 epoch train)
      
#### 4-2) 1 instance → 3 label (Multi-Label Learning)

![](https://i.imgur.com/D9EF3iC.png)

<br>

      (15 epoch train)
      
      Shape (11 Types) 
      Validation Acc: 91.49% 

      Front Color (16 Types) 
      Validation Acc: 91.00% 

      Back Color (16 Types)
      Validation Acc: 90.67%

      → Total Acc: 75.48% 
<br>

## TODO 

### 1. Pill Mask 
      Reference Paper: 
      
      1. Real-world Pill Segmentation based on Superpixel Merge using Region Adjacency Graph
         (Link: http://www.scitepress.org/Papers/2017/61358/61358.pdf)



### 2. Test.py (Testset+test)
