terraform {
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.0" } }
}
provider "aws" { region = "us-east-1" }

resource "aws_instance" "app_server" {
  ami = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.large"
  tags = { Name = "ai-chatbot-app" }
}

resource "aws_rds_cluster" "postgres" {
  cluster_identifier = "ai-chatbot-db"
  engine = "aurora-postgresql"
  database_name = "aichat"
  master_username = var.db_username
  master_password = var.db_password
  skip_final_snapshot = false
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id = "ai-chatbot-redis"
  engine = "redis"
  node_type = "cache.t3.micro"
  num_cache_nodes = 1
}