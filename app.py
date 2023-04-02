from flask import Flask, request, render_template
import pickle
from model import ItemRecommender

app = Flask(__name__)
import pymysql as pms

conn = pms.connect(host="localhost", port=3306,
                   user="root",
                   password="arun",
                   db="ml")
cur = conn.cursor()
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def main():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    
    username = request.form['username']
    password = request.form['password']
    
    cur.execute('SELECT * FROM summa WHERE username = %s AND password = %s', (username, password,))
    account = cur.fetchone()
    if account:
        return render_template("keyword.html")
    else:
        return render_template("login.html", msg='Incorrect username/password!')
    
@app.route("/predict", methods=['post'])
def pred():
    keyword = request.form['genre']
    pred = model.predict(keyword)
    return render_template("recommend.html",data=pred)  
  
if __name__=='__main__':
    app.run(port=5000)
    
    
    
    
    
    
    
    