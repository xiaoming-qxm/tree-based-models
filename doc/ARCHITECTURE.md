## function
1. save_model() <in prototxt>  
2. load_model() <in prototxt>  
3. train()  
4. predict()  
5. score()  


## class
1. <base>  
   FeatureSelectionCriterion()  
　　<derived>  
   Gini()  
   InformationGain()  
   InformationGainRatio()  
2. <base>  
   Model()  
   <derived>
   CART()  
   ID3()  
   C45()  

3. <base>  
   Loss()  
   <derived>  
   SquareLoss()  

4. <base>  
   DataProcessor()  
   <derived>  
   MissedValue()  
   PCA()  
   ZeroMean()  
