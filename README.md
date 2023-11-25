# Premier League Predictions App

## Description
This app provides predictive analysis of upcoming Premier League football matches based on historical data and statistical modeling.

## Features
- Predict match outcomes with statistical accuracy.
- Visualise past results and team performance
- 

## Installation

To set up the project environment:

```bash
git clone https://github.com/your-username/premier-league-predictions-app.git
cd premier-league-predictions-app
pip install -r requirements.txt
```
## Usage
The app is running on Streamlit Community Cloud at https://premier-league-predictor.streamlit.app/

## Data Sources
- Historical results from https://www.football-data.co.uk
- Upcoming fixtures from https://fixturedownload.com
- League table from https://www.footballwebpages.co.uk

## Models
The app uses [Catboost](https://catboost.ai/) for the underlying predictions. There are models for:
- Full time result
- Home Team Score
- Away Team Score

## To Do
- Scheduling
- Deployment
- Flake8 + Black
- More unit tests
- Fix Functions (docs, types)
- Git hooks
- Docs