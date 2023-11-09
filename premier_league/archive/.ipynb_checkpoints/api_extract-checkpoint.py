import requests

url = "https://heisenbug-premier-league-live-scores-v1.p.rapidapi.com/api/premierleague/team"

querystring = {"name":"Liverpool"}

headers = {
    "X-RapidAPI-Key": "0352880adbmshdd982fa19f87c97p12948djsn735a6b2b8176",
    "X-RapidAPI-Host": "heisenbug-premier-league-live-scores-v1.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)


















