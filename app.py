from flask import Flask, request
from Transformer.main import init_transformer, translate

transformer = init_transformer()



app = Flask(__name__)

@app.route('/api/translate/en-vi', methods=['POST'])
def handle_translate():
    if request.method == 'POST':
        data = request.get_json()
        if 'sentence' not in data:
            return "False"
        output = translate(transformer, data['sentence'])
        return output

if __name__ == '__main__':
    app.run()


