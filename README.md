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

### 4. Classifier



#### 4-1) 1 instance → 1 label

![](https://i.imgur.com/yfpsIY4.png)

<br>
      
      1. Shape (11 Types)
      Validation Acc: 91.87%

      2. Front Color (16 Types)
      Validation Acc: 90.23%

      3. Back Color (16 Types)
      Validation Acc: 91.08%
   
      → Total Acc: 75.50%       
      (Each model was trained with 50 epochs)
      
      
#### 4-2) 1 instance → 3 label (Multi-Label Learning)

![](https://i.imgur.com/D9EF3iC.png)

<br>

      1. Shape (11 Types) 
      Validation Acc: 91.55% 

      2. Front Color (16 Types) 
      Validation Acc: 91.70% 

      3. Back Color (16 Types)
      Validation Acc: 90.97%

      → Total Acc: 76.37%    
      (Model was trained with 150 epochs)

<br>

### 5. Uncertainty Model

#### 5-1) 1 instance → 1 label

      1. Shape
      
      Model: Resnet18
      
      Drop-out rate: 0.25
      
      Aleatoric Uncertainty: 0.0085
      
      Epistemic Uncertainty: 0.0001

      Validation Acc: 86.63% (8 epoch Train)
      
      ______________________________________________
      
      2. Color1
      
      Model: Resnet18
      
      Drop-out rate: 0.25
      
      Aleatoric Uncertainty: -
      
      Epistemic Uncertainty: -

      Validation Acc: -
      
      ______________________________________________
            
            
      3. Color2
      
      Model: Resnet18
      
      Drop-out rate: 0.25
      
      Aleatoric Uncertainty: -
      
      Epistemic Uncertainty: -

      Validation Acc: -
      
      → Total Acc: -
      
      <br>
      
#### 5-2) 1 instance → 3 label (Multi-Label Learning)

      1. All (Shape + Color1 + Color2)
      
      Model: Resnet18
      
      Drop-out rate: 0.25
     
      Aleatoric Uncertainty: -
      
      Epistemic Uncertainty: -

      Validation Shape Acc: -
      
      Validation Color1 Acc: -
      
      Validation Color2 Acc: -
      
      → Total Acc: -    

<br>

## TODO 

### 1. Pill Mask 
      Reference Paper: 
      
      1. Real-world Pill Segmentation based on Superpixel Merge using Region Adjacency Graph
         (Link: http://www.scitepress.org/Papers/2017/61358/61358.pdf)



### 2. Test.py (Testset+test)
