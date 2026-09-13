pipeline {
    agent any

    environment {
        PATH = "/opt/homebrew/bin:/Library/Frameworks/Python.framework/Versions/3.13/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        DOCKER_BUILDKIT = "1"
    }

    stages {

        stage('Prepare Git LFS') {
            steps {
                sh '''
                    git config --global filter.lfs.process "/opt/homebrew/bin/git-lfs filter-process"
                    git config --global filter.lfs.smudge "/opt/homebrew/bin/git-lfs smudge -- %f"
                    git config --global filter.lfs.clean "/opt/homebrew/bin/git-lfs clean -- %f"
                    git config --global filter.lfs.required true
                '''
            }
        }

        stage('Checkout') {
            steps {
                deleteDir()

                git branch: 'main',
                    url: 'git@github.com:divyd162-debug/g08-ml-cicd.git'

                sh '''
                    echo "Pulling Git LFS model files..."
                    git lfs pull
                '''
            }
        }

        stage('Verify Model Files') {
            steps {
                sh '''
                    echo "Checking ML model files..."

                    ls -lh models/

                    test -f models/ticket_classifier_best_svm.pkl
                    test -f models/ticket_classifier_calibrated.pkl
                    test -f models/tfidf_vectorizer_best_svm.pkl
                    test -f models/tfidf_vectorizer_calibrated.pkl
                    test -f models/char_tfidf_vectorizer_best_svm.pkl
                    test -f models/char_tfidf_vectorizer_calibrated.pkl

                    echo "All required model files are present."
                '''
            }
        }

        stage('Setup Python') {
            steps {
                sh '''
                    echo "Python version:"
                    python3.13 --version

                    rm -rf ci_venv

                    python3.13 -m venv ci_venv

                    ci_venv/bin/python --version
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    echo "Installing Python dependencies..."

                    ci_venv/bin/pip install --upgrade pip
                    ci_venv/bin/pip install -r requirements.txt

                    echo "Dependencies installed successfully."
                '''
            }
        }

        stage('Run ML Test') {
            steps {
                sh '''
                    echo "Running ML model test..."

                    ci_venv/bin/python test_model.py

                    echo "ML test completed successfully."
                '''
            }
        }

        stage('Docker Check') {
            steps {
                sh '''
                    echo "Checking Docker..."

                    docker --version
                    docker context show
                    docker info

                    echo "Docker is available to Jenkins."
                '''
            }
        }

        stage('Docker Build') {
            steps {
                sh '''
                    echo "Building Docker image..."

                    docker build \
                        -t g08-ml-cicd:${BUILD_NUMBER} \
                        .

                    docker tag \
                        g08-ml-cicd:${BUILD_NUMBER} \
                        g08-ml-cicd:latest

                    echo "Docker image built successfully."

                    docker images g08-ml-cicd
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                    echo "Deploying G-08 ML application..."

                    docker rm -f g08-ml-app-jenkins 2>/dev/null || true

                    docker run -d \
                        --name g08-ml-app-jenkins \
                        -p 5002:5000 \
                        g08-ml-cicd:${BUILD_NUMBER}

                    echo "Application deployed."

                    docker ps --filter name=g08-ml-app-jenkins
                '''
            }
        }

        stage('Deployment Verification') {
            steps {
                sh '''
                    echo "Checking deployed application..."

                    curl --fail \
                         --retry 10 \
                         --retry-delay 2 \
                         http://localhost:5002/

                    echo ""
                    echo "Deployment health check passed successfully."
                '''
            }
        }
    }

    post {
        success {
            echo 'G-08 CI/CD pipeline completed successfully!'
            echo 'Docker deployment is available on port 5002.'
        }

        failure {
            echo 'G-08 CI/CD pipeline failed.'
            echo 'Check the Jenkins console output for the failed stage.'
        }
    }
}