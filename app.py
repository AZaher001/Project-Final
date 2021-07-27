from flask import Flask, render_template, request
import joblib


cut_id = ["Good","Ideal","Premium","Very Good"]
color_id = ["E","F","G","H","I","J"]
clarity_id = ["IF","SI1","SI2","VS1","VS2","VVS1","VVS2"]

model = joblib.load('ai/model.h9')
scaler = joblib.load('ai/scaler.h9')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict_app():
    inp_data = [request.args.get('carat')
    ,request.args.get('depth')
    ,request.args.get('table')
    ,request.args.get('x')
    ,request.args.get('y')
    ,request.args.get('z')]
    cut_dummies  = ['0','0','0','0']
    color_dummies = ['0','0','0','0','0','0']
    clarity_dummies = ['0','0','0','0','0','0','0']
    try:
        cut = cut_id.index(request.args.get('cut_id'))
        cut_dummies[cut]=1
    except:
        pass

    try:
        color = color_id.index(request.args.get('color_id'))
        color_dummies[color] = 1
    except:
        pass

    try:
        clarity = clarity_id.index(request.args.get('clarity_id'))
        clarity_dummies[clarity] = 1
    except:
        pass

    inp_data += cut_dummies
    inp_data += color_dummies
    inp_data += clarity_dummies

    inp_data = [float(n) for n in inp_data]

    price = model.predict(scaler.transform([inp_data]))[0]
    return  render_template('predict.html',price = price)



if __name__ == "__main__":
    app.run()
