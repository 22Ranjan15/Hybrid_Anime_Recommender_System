pipeline{
    agent any

    
    stages{
        stage('Cloning GitHub repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning GitHub repo to Jenkins.......'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/22Ranjan15/Hybrid_Anime_Recommender_System.git']])
                }
            }
        }
    }
}



// pipeline{
//     agent any

//     environment {
//         VENV_DIR = 'venv'
//         GCP_PROJECT = "mlops-projects-457610"
//         GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"

//     }

//     stages{
//         stage('Cloning GitHub repo to Jenkins'){
//             steps{
//                 script{
//                     echo 'Cloning GitHub repo to Jenkins.......'
//                     checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/22Ranjan15/Hotel-Booking-Predictor.git']])
//                 }
//             }
//         }

//         stage('Setting up our Virtual Environment and Installing Dependencies'){
//             steps{
//                 script{
//                     echo 'Setting up our Virtual Environment and Installing Dependencies......'
//                     sh '''
//                     python -m venv ${VENV_DIR}
//                     . ${VENV_DIR}/bin/activate
//                     pip install --upgrade pip
//                     pip install -e .
//                     '''
//                 }
//             }
//         }

//         stage('Building and Pushing Docker Image to GCR'){
//             steps{
//                 withCredentials([file(credentialsId: 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
//                     script{
//                         echo 'Building and Pushing Docker Image to GCR.............'
//                         sh '''
//                         # Copy credentials into the build context
//                         cp ${GOOGLE_APPLICATION_CREDENTIALS} gcp-key.json

//                         export PATH=$PATH:${GCLOUD_PATH}

//                         gcloud auth activate-service-account --key-file=gcp-key.json # Use the copied key

//                         gcloud config set project ${GCP_PROJECT}

//                         gcloud auth configure-docker --quiet

//                         docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

//                         docker push gcr.io/${GCP_PROJECT}/ml-project:latest

//                         # Clean up the copied credentials file
//                         rm gcp-key.json
//                         '''
//                     }
//                 }
//             }
//         }

//         stage('Deploy to Google CLoud Run'){
//             steps{
//                 withCredentials([file(credentialsId: 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
//                     script{
//                         echo 'Deploy to Google CLoud Run.............'
//                         sh '''
//                         # Copy credentials into the build context
//                         cp ${GOOGLE_APPLICATION_CREDENTIALS} gcp-key.json

//                         export PATH=$PATH:${GCLOUD_PATH}

//                         gcloud auth activate-service-account --key-file=gcp-key.json # Use the copied key

//                         gcloud config set project ${GCP_PROJECT}

//                         gcloud run deploy ml-project \
//                             --image=gcr.io/${GCP_PROJECT}/ml-project:latest \
//                             --platform=managed \
//                             --region=us-central1 \
//                             --allow-unauthenticated

//                         # Clean up the copied credentials file
//                         rm gcp-key.json
//                         '''
//                     }
//                 }
//             }
//         }
//     }
// }
