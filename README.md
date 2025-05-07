# ?? Hybrid Anime Recommender System - End-to-End ML Project ??

[![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-orange.svg)]()


## ? Overview

This project is a comprehensive deep learning solution designed to recommend anime titles based on user preferences. By leveraging hybrid recommendation techniques including collaborative filtering and content-based approaches, it aims to provide personalized anime suggestions that match user tastes and viewing history. The project follows the entire ML lifecycle, from data collection and preprocessing to model deployment with an interactive cyberpunk-themed user interface.


## ?? Key Features

* **Personalized Anime Recommendations:** Employs advanced deep learning models to suggest anime titles based on user preference patterns.

* **Hybrid Recommendation Approach:** Combines collaborative filtering and content-based filtering for more accurate recommendations.

* **End-to-End Pipeline:** Covers all stages of the ML pipeline, including data ingestion, processing, model training, and deployment.

* **Cyberpunk-Themed UI:** Visually appealing, responsive user interface with anime-inspired design elements.

* **Scalable Infrastructure:** Built on Google Cloud Platform for scalability and reliability.

* **Web Application Interface:** User-friendly web application built with HTML, CSS and Flask for interacting with the recommendation service.

* **Containerized Application:** Docker is used for containerization, ensuring consistent deployment across different environments.

* **Kubernetes Orchestration:** Configured for deployment on Kubernetes for scalability and high availability.


## ??? Technologies Used
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/2.3.x/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io/)


* **Web Application:** HTML, CSS, Flask
* **Machine Learning:** TensorFlow, scikit-learn, Collaborative Filtering, Content-Based Filtering
* **Data Storage:** Google Cloud Storage
* **Containerization:** Docker
* **Orchestration:** Kubernetes
* **Deployment:** Google Cloud Platform (GCP)

## ?? Project Structure
This repository implements a production-grade machine learning application with a modular, maintainable architecture:
```
HYBRID ANIME RECOMMENDER SYSTEM/
??? .dvc/
??? .dvcignore
??? .gitignore
??? artifacts/
?   ??? model/
?   ?   ??? model.h5
?   ??? processed/
?   ?   ??? anime_df.csv
?   ?   ??? rating_df.csv
?   ?   ??? synopsis_df.csv
?   ??? raw_data/
?       ??? anime_with_synopsis.csv
?       ??? anime.csv
?       ??? animelist.csv
??? config/
?   ??? init.py
?   ??? config.yaml
?   ??? paths_config.py
??? custom_jenkins/
?   ??? Dockerfile
??? notebooks/
?   ??? experiments.ipynb
??? pipeline/
?   ??? init.py
?   ??? prediction_pipeline.py
?   ??? training_pipeline.py
??? src/
?   ??? init.py
?   ??? components/
?   ?   ??? init.py
?   ?   ??? base_model.py
?   ?   ??? data_ingestion.py
?   ?   ??? data_processor.py
?   ?   ??? model_trainer.py
?   ??? exception.py
?   ??? logger.py
??? static/
?   ??? images/
?   ?   ??? favicon.ico
?   ??? style.css
??? templates/
?   ??? index.html
??? utils/
?   ??? init.py
?   ??? helpers.py
?   ??? utils.py                             
??? app.py                                
??? deployment.yaml                      
??? Dockerfile                            
??? Jenkinsfile                           
??? project_structure.py                 
??? README.md                            
??? requirements.txt                      
??? setup.py                              
??? Test.py                               
```

## ?? Model Development and Selection
The recommendation system uses a hybrid approach combining:

1. **Collaborative Filtering:** Identifies similar users and recommends anime titles they've enjoyed

2. **Content-Based Filtering:** Recommends anime based on similar features (genre, studio, theme)

3. **Hybrid Model:** Combines both approaches for more accurate recommendations


Key highlights from the model development process:
* Data preprocessing included handling missing values and encoding categorical features

* The system addresses the cold-start problem by incorporating content-based recommendations

* Evaluation metrics included precision, recall, and user satisfaction metrics

## ?? Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:

* Python 3.10
* Google Cloud SDK
* Docker
* Kubernetes (optional for local development)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Hybrid-Anime-Recommender-System.git
   cd "Hybrid Anime Recommender System"
   ```

2. Set up your Google Cloud credentials:
   > ?? Follow the [Google Cloud documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) for instructions.

3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   * Create a [`.env`](.env ) file and add necessary environment variables like:
     * GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_cloud_credentials.json

### Usage

1. To run the Flask web application locally:
   ```bash
   python app.py
   ```
   The application will be accessible at **`http://localhost:5000`**

2. To build and run the Docker container:
   ```bash
   docker build -t anime-recommender-app .
   docker run -p 5000:5000 anime-recommender-app
   ```
   The application will be accessible at **`http://localhost:5000`**.

3. To deploy to Kubernetes:
   ```bash
   # Create a secret for GCP credentials
   kubectl create secret generic gcp-credentials --from-file=key.json=/path/to/your/gcp-key.json
   
   # Apply the deployment
   kubectl apply -f deployment.yaml
   ```

## ?? Cloud Deployment

The application is deployed on Google Cloud Platform. Specific services used include:

- **Google Cloud Storage:** Used for storing and accessing data files and models

- **Google Kubernetes Engine (GKE):** Used to deploy and orchestrate the containerized application

- **Google Container Registry (GCR):** Used to store and manage Docker container images

### Deployment Architecture

- The Flask application is containerized using Docker and pushed to **Google Container Registry**

- **Google Kubernetes Engine** deploys the containerized application using the configuration in [`deployment.yaml`](deployment.yaml )

- The application communicates with **Google Cloud Storage** to access anime data and trained models

- Load balancing and scaling are handled automatically by Kubernetes

## ?? Demo

- ?? Screenshots of the anime recommender interface:
  - Homepage with cyberpunk-themed UI
  - Results page showing personalized anime recommendations
  - Mobile-responsive design

- ?? Demo: [Anime Recommender System](https://drive.google.com/file/d/10f1Jncpyh7YRwu4WcqtnV06DE1L9x3ha/view?usp=sharing)

## ?? Performance

The hybrid recommendation system achieves:

- **Recommendation Relevance:** High match between recommended anime and user preferences

- **Response Time:** Fast recommendation generation (typically under 2 seconds)

- **Cold Start Handling:** Effective recommendations even for new users with limited history

The system continuously improves as more user data becomes available.

## ?? Contributions

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## ?? License

This project is licensed under the MIT License.

## ?? Contact

### Your Name
- **Email:** [ranjandasbd22@gmail.com](ranjandasbd22@gmail.com)
- **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/das-ranjan22/)
- **GitHub:** [22Ranjan15](https://github.com/22Ranjan15)

---
*Feel free to reach out for questions, collaborations, or feedback about this project!*