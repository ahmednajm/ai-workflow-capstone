# IBM AI Enterprise Workflow Capstone

Files and example of solution for the IBM AI Enterprise Workflow Capstone project. 

The IBM AI Workflow capstone project is part of the evaluation for the IBM AI Workflow Coursera specialization. In short, it simulates the full cycle of a data science project, and consists of understanding the data and the business opportunity related to a fictitious video streaming company; comparing models and choosing the adequate one; and developing a solution ready for deployment, including creating an API with the needed endpoints and building a Docker image to containing the solution. 

The dataset presented consists of months worth of data coming from said streaming company, including information about views and revenue, as well as the countries originating those. The main business opportunity stated is to try to forecast revenue values.

The project is then composed of three parts. The first one involves investigating the data, searching for relationships therein, and articulating findings using visualizations. The second one is comprised of trying different modelling approaches, and selecting one to be used for deployment, based on the observations. Lastly, the third one consists of building a draft version of the API and serve it via a Docker image.

# Project Execution Instructions

## Using the model_module directly

Run server_side/model_module.py [-m={dev, prod}] [--task={train, load, predict}] [-model {rf,et}] [-s {ss,rs}] -c 'country'

### Options:    
--mode {dev, prod}, default='prod',

  train either a development or production model. Omitting this implies loading an already-trained model

--task {train, load, predict} 

  choosing whether we are training a model, or predicting from an already-trained model, or just loading existing models
  
-model {rf,et}, --model {rf,et}

  (rf) RandomForestRegressor or (et) ExtraTreesRegressor (default)
  
-s {ss,rs}, --scaler {ss,rs}

  (ss) StandardScaler or (rs) RobustScaler (default)
  
-c country 

  The country to predict revenue for (default: 'World')
       
## Unit tests
### Model Tests
Run python3 -m unittest unittests/server_side_ModelTests.py
### Logger Tests
Run python3 -m unittest unittests/server_side_LoggerTests.py
### API Tests
Run python3 -m unittest unittests/client_side_ApiTests.py

## Performance monitoring
Run python3 server_side/reading_metrics.py

## API DOCUMENTATION

| Request type    | Key            | Description                                        |
|-----------------|----------------|----------------------------------------------------|
| `/train`        | `query`        | Query for model, must be a dict containing 'country', 'model', 'scaler' as keys, with their values as strings. |
|                 | `mode`         | Training task - dev or prod                        |
|                 |                |                                                    |
| `/predict`      | `query`        | Query for model, must be a dict containing 'country', 'year', 'month', 'day' as keys, with their values as strings. |
|                 | `mode`         | Predict task - dev or prod                         |
|                 |                |                                                    |
| `/logs`         | `filename`     | Name of log file to retrieve                       |


You can run the server with:

client_side/api.py  or  client_side/api.py -d

## Docker Image
The Docker image for this project is hosted on Docker Hub :

https://hub.docker.com/r/ahmednajm/aavail-app

You can pull it by running :

docker pull ahmednajm/aavail-app:latest

Alternatively, you can build the image from the Dockerfile by running :

docker build -t rdpb/ibm-ai-capstone.

After pulling or building, you can run the image as a container:

docker run -d -p 8080:8080 ahmednajm/aavail-app:latest