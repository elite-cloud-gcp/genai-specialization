from flask import Flask, render_template

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    """
    Renders the main page which includes the Dialogflow messenger.
    """
    return render_template('index.html')

if __name__ == '__main__':
    # Run the app on all available interfaces on port 8080
    app.run(debug=True, host='0.0.0.0', port=8080)