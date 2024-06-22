aws ecr get-login-password --region "us-west-1" | docker login --username AWS --password-stdin "723854145424.dkr.ecr.us-west-1.amazonaws.com"
docker compose up
