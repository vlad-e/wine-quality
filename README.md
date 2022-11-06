Work in progress

# Classifying wines based on their physico-chemical proprieties
## Mid-term project for DataTalks - multiclass classification problem


Hello, thanks for visiting this small project.

Dataset extracted from https://archive.ics.uci.edu/ml/datasets/wine+quality

How to:  
         1. Build  Docker image using ``` docker build -t wine-quality .```

         2. Deploy image using  ``` docker run -it --rm -p 9696:9696 wine-quality ```
         
         3. Waitress will be listening on port 96969 so go ahead and test the service using ``` python predict-test.py ``` in another terminal
         
Future work will be done on the data and the model choice.
