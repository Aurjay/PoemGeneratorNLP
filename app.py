from flask import Flask, render_template, request
from keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')
standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
  tokenizer = Tokenizer()
  data = open('dataforpoem.txt').read()
  corpus = data.lower().split("\\n")
  tokenizer.fit_on_texts(corpus)

  json_file = open('model.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights('modeling.h5')
  print("loaded model from disk")
  if request.method == 'POST':
      seed_text = "STARTER-Hello you beautiful creature."
      next_words = 100

      for _ in range(next_words):
          token_list = tokenizer.texts_to_sequences([seed_text])[0]
          token_list = pad_sequences([token_list], maxlen=21, padding='pre')
          predicted = loaded_model.predict_classes(token_list, verbose=0)
          output_word = ""
          for word, index in tokenizer.word_index.items():

              if index == predicted:
                  output_word = word
                  break
          seed_text += " " + output_word


      return render_template('index.html',prediction_text="The generated poem is: {}".format(seed_text))

if __name__=="__main__":
    app.run(debug=True)

