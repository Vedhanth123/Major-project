import requests

def get_geolocation():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        city = data.get("city", "Unknown")
        country = data.get("country", "Unknown")
        return city, country
    except Exception as e:
        print(f"[ERROR] Geolocation failed: {e}")
        return "Unknown", "Unknown"
if __name__ == "__main__":
    city, country = get_geolocation()
    print(f"You are in {city}, {country}")
