from flask import Flask, request, render_template, flash, jsonify
import run
from run import run
#import os
#import sys

#sys.path.append('../../PycharmProjects')
#print(sys.path)

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def handle_data():
    try:
        if (request.method == 'POST'):
            #jsonify(data)
                return "airplane&angel&arm&banana&bell"

        else:
            return "airplane&angel&arm&banana&bell"
                #"Hello World - you sent me a GET " + str(request.values)
    except Exception as e:
        flash(e)
        return "Error" + str(e)


@app.route("/send", methods=['POST','GET'])
def return_probables():
    try:
        if (request.method == 'POST'):
            #os.system("python run.py");
                return run()

        else:
            return "airplane&angel&arm&banana&bell"
                #"Hello World - you sent me a GET " + str(request.values)
    except Exception as e:
        flash(e)
        return "Error" + str(e)



@app.route("/home", methods=['GET'])
def homepage():
    return render_template("index.html")




if __name__ == '__main__':
    app.run(host= '0.0.0.0')

