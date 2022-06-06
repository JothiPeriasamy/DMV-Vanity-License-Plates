# FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-6
FROM python:3.7.4

EXPOSE 8080

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r Utility/DMV_requirements.txt



# Sets up the entry point to invoke the trainer.
CMD streamlit run --server.port 8080 --server.enableCORS false --global.developmentMode=false --server.headless=true DMV_App.py