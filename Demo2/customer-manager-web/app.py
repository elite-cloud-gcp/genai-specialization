import json
from flask import Flask, send_from_directory, jsonify, request

# File to store customer data
DATA_FILE = "customers.json"

# Load customer data from JSON file
def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:  # If file doesn't exist, start with empty data
        return {}

# Save customer data to JSON file
def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# Read initial customer data
customers = load_data()

# Create Flask web application
app = Flask(__name__, static_url_path='', static_folder='static')

# Serve the main webpage
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# API endpoint to get all customers
@app.route('/api/customers', methods=['GET'])
def get_customers():
    return jsonify(list(customers.values())), 200

# API endpoint to get a specific customer by ID
@app.route('/api/customers/<int:customer_id>', methods=['GET'])
def get_customer(customer_id):
    print("Received request for customer ID:", customer_id) 
    for key, value in customers.items():
        if value.get("customer_id") == customer_id:
            print("Customer found:", value)
            return jsonify(value), 200
    return jsonify({"message": "Customer not found"}), 404

# API endpoint to update a customer by ID
@app.route('/api/customers/<int:customer_id>', methods=['PUT'])
def update_customer(customer_id):
    updated_customer = request.json

    # Find and update the customer (if they exist)
    for key, existing_customer in customers.items():
        if existing_customer.get("customer_id") == customer_id:
            existing_customer.update({k: v for k, v in updated_customer.items() if k != "customer_id"})
            save_data(customers)  # Save changes to file
            return jsonify(existing_customer), 200
    return jsonify({"message": "Customer not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 
