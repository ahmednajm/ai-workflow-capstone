
import os, re, argparse
from flask import Flask, request, jsonify, send_from_directory
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from server_side.model_module import model_train, model_predict
from server_side.model_module import CLEAN_DATA_DIR, MODEL_PARAM_GRID, LOG_DIR


# Create server (flask) and Dash app
server = Flask(__name__)
#app = create_dash_app(server)

# 
#API_LOG_DIR = os.path.join(os.path.dirname(__file__), '.', "api_logs")
#if not os.path.isdir(API_LOG_DIR):
#    os.mkdir(API_LOG_DIR)
    
# Configure logger to log API-related events
#logger.add(os.path.join(API_LOG_DIR, "api_log_{time}.log"), 
#           format="{time:YYYY-MM-DD HH:mm} | {level} | {message}",
#           level="INFO"
#          )

# Allow external services to verify if the Flask app is up and running.
@server.route('/ping', methods=['GET', 'POST'])
def ping():
    return jsonify({
        'status': 200,                   
        'message': 'Ping successful!' 
    })

# Get model instance by name in the Train API Endpoint
def get_model_instance(model_name):
    # Map model names to actual model instances
    model_mapping = {
        "rf": RandomForestRegressor(random_state=42),
        "et": ExtraTreesRegressor(random_state=42),
    }
    return model_mapping.get(model_name)


# Get scaler instance by name in the Train API Endpoint
def get_scaler_instance(scaler_name):
    scaler_mapping = {
        "ss": StandardScaler(),
        "rs": RobustScaler(),
    }
    return scaler_mapping.get(scaler_name)


# Train API Endpoint
@server.route('/train', methods=['POST'])
def train():
    """
    A Endpoint for model training based on specified parameters.
    Expects JSON input with 'scaler', 'model', 'version', 'mode', 'country' will be 'all' if not provided in the request...
    A cmd line example on terminal to make the call :
    curl -X POST http://127.0.0.1:5000/train \    
    -H "Content-Type: application/json" \
    -d '{"country": "United Kingdom", "version": 6, "mode": "prod"}'
    """
    try:
        # Extract JSON data from the request
        train_request = request.get_json()
    
        # Check for request data
        if not train_request:
            return jsonify({"ERROR: API (train): did not receive request data"}), 400

        # Extract required parameters
        clean_data_dir = train_request.get('clean_data_dir', CLEAN_DATA_DIR)
        country = train_request.get('country', 'all') # training a model for all countries if country not provided
        version = train_request.get('version')
        model_name = train_request.get('model', 'et') # default: ExtraTreesRegressor
        model_param_grid = train_request.get('model_param_grid', MODEL_PARAM_GRID)
        model_scaler_name = train_request.get('model_scaler', 'rs')
        mode = train_request.get('mode', 'prod')  # The default mode is 'prod' if mode not provided
        
        # Validate required parameters
        if not version : # or not country or not version or not model_name :
            missing_fields = [field for field in ['clean_data_dir', 'country', 'version', 'model', 'model_scaler'] if not train_request.get(field)]

            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Map model name to model instance (you can customize this mapping)
        model = get_model_instance(model_name)
        if model is None :
            return jsonify({"error": f"Model '{model_name}' is not recognized."}), 400

        # Map scaler name to scaler instance if provided
        model_scaler = get_scaler_instance(model_scaler_name)
        if model_scaler is None :
            return jsonify({"error": f"Scaler '{model_scaler_name}' is not recognized."}), 400

        # Call the model training function
        print(f"\n... Training the version {version} of the [ {model_scaler} + {model} ] model for '{country}' in '{mode}' mode...")
        train_output = model_train(clean_data_dir, country, version, model, model_param_grid, model_scaler, mode)
        print("\nTraining completed successfully\n")

        # Log the start of the training process
        #logger.info(f"Version {version}. The parameters grid: {model_param_grid}. The pipe:[ {model_scaler} + {model} ]. The country: {country}. The mode:'{mode}'. Evaluation metrics: {train_output['evaluation metrics']}. The best parameters: {train_output['best model parameters']}.")

        # Return a success message and any relevant training info
        return jsonify({
            "status": "Training completed successfully",
            "training_output": train_output
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
            

# Predict API Endpoint
@server.route('/predict', methods=['POST'])
def predict():
    """
    A Endpoint for revenue prediction based on specified parameters.
    Expects JSON input with 'year', 'month', 'day', country will be 'all' if not provided in the request...
    A cmd line example on terminal to make the call :
    curl -X POST http://127.0.0.1:5000/predict \
    -H "Content-Type: application/json" \
    -d '{
          "clean_data_dir": "path/to/data",
          "country": "United Kingdom",
          "year": 2024,
          "month": 11,
          "day": 8,
          "mode": "prod"
        }'
    """
    try:
        # Extract JSON data from the request
        predict_request = request.get_json()
        
        # Check for request data
        if not predict_request:
            return jsonify({"ERROR: API (predict): did not receive request data"}), 400

        # Extract required parameters
        clean_data_dir = predict_request.get("clean_data_dir", CLEAN_DATA_DIR)
        country = predict_request.get("country", 'all') # predicting from a model trained for all countries if country not provided
        year = predict_request.get("year")
        month = predict_request.get("month")
        day = predict_request.get("day")
        mode = predict_request.get("mode", "prod")  # The default mode is 'prod' if mode not provided
        
        # Validate parameters
        if not all([year, month, day]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Call model_predict
        prediction_result = model_predict(clean_data_dir, country, year, month, day, mode, models_dict= None)
        
        # Check if model_predict returned a list (indicating date is out of range)
        if isinstance(prediction_result, tuple) and len(prediction_result) == 3:
            start_date, end_date, nearest_date = prediction_result
            return jsonify({
                "error": f"The specified date is out of the available data range",
                "hint": f"Please provide a date within the range [{start_date} - {end_date}] or adjust to {nearest_date}, the nearest available date"
            }), 400
        
        # Return a success message and the requested prediction
        return jsonify({
            "status": "Prediction completed successfully",
            f"expected revenue over the next 30 days for {country}": f"{prediction_result:,.0f}"
                      }), 200

    except Exception as e:
        # Log the error
        logger.error(f"Error in /predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500        
        
# List of Logs API Endpoint
@server.route('/logs/',methods=['GET'])
def list_logs():
    """
    API endpoint to get list of logs
    """
    try :
        if not os.path.isdir(LOG_DIR):
            print("ERROR: API (log): cannot find api logs dir")
            return jsonify([])
    
        return jsonify(os.listdir(LOG_DIR)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500        


@server.route('/logs/<filename>',methods=['GET'])
def logs(filename):
    """
    API endpoint to get logs
    """
    try:
        if not re.search(".csv",filename):
            print(f"ERROR: API (log): the requested file is not a .csv file: {filename}")
            return jsonify([])

        if not os.path.isdir(LOG_DIR):
            print("ERROR: API (log): cannot find the logs directory")
            return jsonify([]), 400

        file_path = os.path.join(LOG_DIR,filename)
        if not os.path.exists(file_path):
            print(f"ERROR: API (log): the requested file could not be found: {filename}")
            return jsonify([]), 400

        return send_from_directory(LOG_DIR, filename, as_attachment=True)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':

    # parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = ap.parse_args()

    if args.debug:
       server.run(debug=True, port=8080)
    else:
       server.run(host='0.0.0.0', threaded=True, port=8080)
        
    




        
    

    



