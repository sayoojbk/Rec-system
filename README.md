# Recommendation Microservices

**Note:** 
    * You will only need docker installed on your computer to run this app

## Git Steps
1. Fork Branch
2. Open terminal and clone **forked branch**: `git clone https://github.com/<YOUR USERNAME>/recommendation.git`
3. Go inside point-system directory: `cd recommendation`
3. Add upstream repo: `git remote add upstream https://github.com/fcgl/recommendation.git`
4. Confirm that you have an origin and upstream repos: `git remote -v`

## Build & Run App

This build should work for both macOS and Linux

1. Download docker for your operating system
2. From project root run the following commands:
    * **Build:** `docker build -t recommendation .`
    * **Run:** `docker run --name rec -p 5000:5000 recommendation`

## Health Endpoint

Confirm everything was ran correctly by going to the following endpoint: 
    * http://localhost:5000/health/v1/marco
