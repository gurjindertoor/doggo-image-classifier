# Doggo Image Classifier 🐶

## Description
This project utilizes a deep learning model trained on a large dataset of over 10,000 images across 120 different classes of dog breeds. Leveraging the power of TensorFlow, TensorFlow Hub, Pandas, NumPy, and Python, it's capable of identifying a wide variety of dog breeds from images.

Additionally, a user-friendly front-end interface has been developed using Flask, allowing users to easily interact with the model and classify images of their own dogs.

## Prerequisites
Before you begin, ensure you have met the following requirements:

* Python 3.x
* TensorFlow 2.x
* TensorFlow Hub
* Flask
* NumPy
* Pandas
* wtforms
* Pillow

You can install any missing requirements with pip:
  ```bash
  python -m venv env
  ```
  ```bash
  .\env\Scripts\activate
  ```
  ```bash
  pip install tensorflow tensorflow_hub numpy pandas requests flask flask_uploads wtforms Pillow
```
## Getting an API Key
This project requires an API key from [API Ninjas](https://api-ninjas.com/). To obtain it, please follow the steps below:

1. Create an account at [API Ninjas](https://api-ninjas.com/).

2. Once you've successfully created your account, navigate to 'My Account'.

3. Under 'API Key' on the top left, click 'Show API Key'. This is the key you'll need for the project.

Remember to keep this key safe, as it is unique to your account and grants access to the API.


## Usage
To use the Doggo Image Classifier, follow these steps:

1. Clone this repository:
  ```bash
  git clone https://github.com/gurjindertoor/doggo_image_classifier.git
  ```

2. Optionally change the SECRET_KEY:
  ```bash
  app = Flask(__name__)
  app.config["SECRET_KEY"] = "doggo_classifier"
  ```
  
3. Input your own API key
  ```bash
  @app.route("/breed_info/<breed>")
  def get_breed_info(breed):
    api_url = "https://api.api-ninjas.com/v1/dogs?name={}".format(breed)
    response = requests.get(
        api_url, headers={"X-Api-Key": "YOUR-API-KEY"}
    )
   ```

4. Navigate into the project directory:
  ```bash
  cd doggo_image_classifier
  ```
 
5. Run the Flask application:
  ```
  bash
  python app.py
  ```
  
6. Open your web browser and navigate to:
  ```bash
  https://localhost:3000
  ```
  
## Model Training
The machine learning model was trained using TensorFlow and TensorFlow Hub, taking advantage of transfer learning to expedite the process and increase the accuracy of the final model.

The training data consisted of over 10,000 images from 120 different dog breeds. Each image was preprocessed and then used to train the model. The process included the stages of loading and preprocessing the data, building and training the model, and finally evaluating and saving it.

## Contributing
We welcome any contributions! To do so, please follow these steps:

1. Fork this repository.
2. Create a branch: git checkout -b <branch_name>.
3. Make your changes and commit them: git commit -m '<commit_message>'
4. Push to the original branch: git push origin <project>/<location>
5. Create the pull request.
  
## Contact
If you want to contact me, you can reach me at gurjindertoor1@gmail.com

## License
This project uses the following license: MIT.
  
## Images

![doggoclassifier1](https://github.com/gurjindertoor/doggo_image_classifier/assets/78512847/65975b3e-9734-4eea-bcb2-25961876d99d)

![doggoclassifier2](https://github.com/gurjindertoor/doggo_image_classifier/assets/78512847/1ab70601-0e75-49e6-be91-09174d9c2058)


