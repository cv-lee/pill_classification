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

### 4. Shape (11 Types) Train 

      Validation Acc: 87% (2 epoch train)

### 5. Front Color (16 Types) Train 

      Validation Acc: 80% (4 epoch train)

### 6. Back Color (16 Types) Train 

      Validation Acc: 
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
