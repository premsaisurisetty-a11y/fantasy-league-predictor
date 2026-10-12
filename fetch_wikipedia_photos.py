import urllib.request
import urllib.parse
import json
import time

players = [
    "Virat Kohli", "Rohit Sharma", "Jasprit Bumrah", "MS Dhoni", "Shubman Gill",
    "Hardik Pandya", "Sanju Samson", "KL Rahul", "Suryakumar Yadav", "Rashid Khan",
    "Rishabh Pant", "Ravindra Jadeja", "Shreyas Iyer", "Mohammed Siraj", "Mohammed Shami",
    "Arshdeep Singh", "Kuldeep Yadav", "Axar Patel", "Bhuvneshwar Kumar", "Deepak Chahar",
    "Pat Cummins", "Trent Boult", "Jos Buttler", "David Miller", "Quinton de Kock",
    "Heinrich Klaasen", "Ruturaj Gaikwad", "Rinku Singh", "Shivam Dube", "Shardul Thakur",
    "Ravi Bishnoi", "Varun Chakravarthy", "Ajinkya Rahane", "Cameron Green (cricketer)", "Jofra Archer",
    "Kagiso Rabada", "Mitchell Santner", "Ishan Kishan", "Tilak Varma", "Rajat Patidar",
    "Yash Dayal", "Devdutt Padikkal", "Phil Salt (cricketer)", "Tim David", "Will Jacks",
    "Mayank Markande", "Jaydev Unadkat", "Josh Hazlewood", "Keshav Maharaj", "Noor Ahmad",
    "Prabhsimran Singh", "Nitish Kumar Reddy", "Prasidh Krishna", "Mahipal Lomror",
    "Sameer Rizvi", "Naman Dhir"
]

results = {}

headers = {
    'User-Agent': 'FantasyScoreAI-App/2.0 (student_ml_project@college.edu)'
}

for raw_name in players:
    clean_name = raw_name.replace(" (cricketer)", "")
    title = raw_name.replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        res = urllib.request.urlopen(req, timeout=5)
        data = json.loads(res.read().decode('utf-8'))
        thumb = data.get('thumbnail', {})
        img_url = thumb.get('source', '')
        page_id = str(data.get('pageid', ''))
        
        if img_url:
            results[clean_name] = {
                "id": page_id,
                "url": img_url,
                "title": data.get("title", clean_name)
            }
            print(f"✓ {clean_name}: {img_url[:80]}...")
        else:
            print(f"✗ {clean_name}: No thumbnail found")
    except Exception as e:
        print(f"Error {clean_name}: {e}")
    time.sleep(0.1)

with open("verified_wikipedia_images.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nTotal verified photos: {len(results)} / {len(players)}")
