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