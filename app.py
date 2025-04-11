from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# In-memory user "database"
users = [{"id": 1, "name": "Aditya"}]

# Serve a webpage
@app.route('/')
def home():
    return render_template("index.html")

# REST API: Get all users
@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(users)

# REST API: Add a new user
@app.route('/api/users', methods=['POST'])
def add_user():
    data = request.get_json()
    new_user = {
        "id": len(users) + 1,
        "name": data['name']
    }
    users.append(new_user)
    return jsonify(new_user), 201

if __name__ == '__main__':
    app.run(debug=True)
