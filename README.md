# End-to-End Image Classification App

This repostitory contains my end-to-end image classification app using Streamlit, Azure, and GitHub Actions.

## Objective 

The objective of this project is to successfully follow an end-to-end machine learning development and deployment cycle. The focus was not to create a stellar and accurate machine learning model, but to display a simple application that can take user input that can be both hosted and monitered in the cloud. The model is a simple Convolutional Neural Network (CNN) from the Keras library. This project was inspired by [this blog](https://medium.com/analytics-vidhya/mlops-end-to-end-machine-learning-pipeline-cicd-1a7907698a8e) and this [YouTube video](https://www.youtube.com/watch?v=g687fRBNNGo&list=PLzKPJ17mpgj1satmcggLnittevapVNdgM&index=5). The application can be viewed [here](https://imageclassificationapp-fvhmaxbsb8cebedj.centralus-01.azurewebsites.net).

## Dataset

This dataset is a collection of around 25k images consisting of 6 categories: buildings, forest, glacier, mountain, sea, street. The data is already split into training, test, and prediction sets. I downloaded the dataset through [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data?select=seg_pred) and the files can also be accessed through this repository.

## Tools

The main tools used in this project were Python, TensorFlow, Streamlit, Docker, Azure, GitHub Actions.  

## Notebook

### Data Preprocessing

After loading in the data, I used an 'ImageDataGenerator()' to rescale the images and load the training and test sets into batches that the CNN anticipates as input. 

```python
train_datagen = ImageDataGenerator(rescale=1/255.,
                                   rotation_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(150,150),
                                               batch_size=32,
                                               class_mode='categorical')
test_data = val_datagen.flow_from_directory(test_dir,
                                           target_size=(150,150),
                                           batch_size=32,
                                           class_mode='categorical')
```

### Model Training
With the data prepared, the CNN model can now be built. This architecture consists of 3 convolutional layers, 3 pooling methods and 2 hidden layers with the last as an output. 

```python
model_1 = Sequential([
  Conv2D(16, 3, padding='same', activation='relu', input_shape=(150,150,3)),
  MaxPool2D(),
  Conv2D(32, 3, padding='same', activation='relu'),
  MaxPool2D(),
  Conv2D(64, 3, padding='same', activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(len(class_names), activation='softmax')
])

model_1.compile(loss="categorical_crossentropy",
              optimizer=Adam(),
              metrics=['accuracy'])
model_1.summary()
```

The model summary is shown here:

[model summary](images/)

After building the model, it can now be trained and validated. This training only utilized 4 epochs for training as the goal was not to create a strong performing model. This project can be advanced by adding more layers to the CNN model as well as adding more epochs to the training step.

```python
history_1 = model_1.fit(train_data,
                    epochs=4,
                    batch_size=32,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))

model_1.evaluate(test_data)
```

Model evaluation:

[model eval](images/)

This is sufficient enough to move on to plotting validation graphs.  

### Model Validation

The model can now be validated with the test dataset and this can be viewed through plots using the matplotlib library. These are the plots for validation loss and accuracy:

[val loss](images/)


[val accuracy](images/)

With the model looking passable enough to demo in an app, it is time to save the model using

```python
model_1.save('./model.h5')
```
This saves the model as an .h5 file in the local directory for use in the Streamlit application.

## Streamlit Application

### App Framework
The main framework is a very simple visual application to use the model. The flow is for the application to load the model, cache it for subsequent use, accept user input through pasting a url of a picture that follows the 6 classes, then decode the image and display both the predicted class and the image used. This process can be repeated for as long as the user desires. I included a try-except block to handle a schema error with the requests libary. 

The Streamlit application looks like this and this is what will be hosted on Azure:

[app](images/)

### Dockerfile

With the app framework complete, I created a Dockerfile that will later be used in Azure to build a docker image that contains the app file, saved model, and the requirements.txt file needed to successfully run the application. This was the Dockerfile used for this project:

```dockerfile
# lightweight python
FROM python:3.9-slim

RUN apt-get update

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

EXPOSE 8501

RUN ls -la $APP_HOME/

# Install dependencies
RUN pip install -r requirements.txt

# Run the streamlit on container startup
CMD [ "streamlit", "run","--server.enableCORS","false","app.py" ]
```
## Azure

### Container Registry

With the Dockerfile created and Docker image built and recognized in Docker Hub, I was able to start the process to begin hosting the application in Azure. I created a resource group specifically for this project and accepted a majority of the default settings when creating the registry.

### Pushing Docker Image to Azure

The container is now ready for the Docker image to be built using the Azure Container Registry url and pushed to the registry. This was done using 3 terminal commands while enduring Docker Hub was running in the background:

```powershell
docker build -t azure-container-url.io/app-name:latest .

docker login azure-container-url

docker push azure-container-url.io/app-name:latest
```

The `azure-container-url.io` is the url that was given to me when creating the container registry, this will be different if replicated. The `app-name` is customizable, I chose `imageclassificationapp`. When replicating, this will be different and dependent on the name chosen when created. The login command will prompt the login information for Azure and requires a password generated by the Access Keys in the registry. The final push command allows the Docker image to be recognized when creating the Web App. 

### Creating Azure Web App

Now that the Container Registry is created and the Docker image has been pushed to the registry, the Web App can now be created and deployed. The steps for this were relatively straightforward. 
- The same resource group as the Container Registry was selected.
- I chose `imageclassificationapp` as the name of the application and to be consistent with the docker commands.
- The "Container" option was selected in the Publish field and accepted default options for the rest of the first page
- In the Container section of the setup, my created registry was chosen and the image name was manually added.

With these settings, the web app was able to deploy successfully!

### Enabling GitHub Actions

The final step was to incorporate GitHub Actions for monitoring. From the Web App screen in Azure, there is an option for GitHub Actions once continuous depolyment is enabled. After syncing my GitHub account, I was able to select this repository and saved the configuration. The workflow file was created instantly and I was able to once again view a successful deployment.

## Conclusion

This project gave me great insight into the process of MLOps. I was able to dive right into tools such as Docker and Azure and I'm excited to learn more as I tackle more difficult projects. I did run into some issues along the way which I hope to understand more as I keep developing skills in model deployment. Azure gave me some issues selecting the region closest to me, I assume this because I am on the free tier as of the publication of this project. When incorporating GitHub Actions, the final url was not available as it was giving me a warning involving secrets. As far as I knew, the url did not contain secrets, but as I work to understand yaml files more, maybe this issue will get addressed. 
