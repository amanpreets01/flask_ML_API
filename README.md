# flask_ML_API
A simple flask ML API for classifying Iris dataset as an example to use for websites 

Steps : 
1. A simple interactive API can be useful because we usually develop models on Python and website is then required to be built on the 
    entire any Python framework
  
2.If an API built given data through Javascript and gets the classified result ,then it is solved



Processes:

    API ref : /get_type/sl/

1.Run the train_classifier.py
  The model will be saved with name 'iris_classifier.sav'
  
2.Run the flask globally or in virtual environment as you wish

    Note:Dont forget to SET FLASK_APP=app.py

3.Simply run "flask run"

4 .Then either use curl to request or use Postman : https://www.getpostman.com/

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Another approach is when we want to use any Neural Net as a classifier

    API ref : /get_type/tp

  Logic is the same : save_weights + build_architeture_at_reference + load_weights ------> pass the parameters onto model for classifying


1.Run the IrisTorchClassifier.py
  The model will be saved with name 'IrisTorchClassifier.pt'
  
2.Run the flask globally or in virtual environment as you wish

    Note:Dont forget to SET FLASK_APP=app.py

3.Simply run "flask run"

4 .Then either use curl to request or use Postman : https://www.getpostman.com/
