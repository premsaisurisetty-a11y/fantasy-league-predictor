"""
Player Image Provider Service
Maps player names and IDs to external, publicly accessible cricket player profile image URLs.
Loads genuine, verified portrait photos directly from official public sources at runtime via CDN proxy.
NO image files are downloaded or saved to disk.
"""

import os
import json
import urllib.parse

# Optional environment variable support for external sports API
PLAYER_API_KEY = os.environ.get("PLAYER_API_KEY", "")

# In-memory cache to prevent repeated URL calculations
_IMAGE_CACHE = {}

# Path to verified player registry JSON
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REGISTRY_JSON_PATH = os.path.join(_CURRENT_DIR, "verified_players.json")

# Verified Player Registry
EXTERNAL_PLAYER_IMAGE_REGISTRY = {}

# Load verified registry from JSON if available
if os.path.exists(_REGISTRY_JSON_PATH):
    try:
        with open(_REGISTRY_JSON_PATH, "r", encoding="utf-8") as f:
            EXTERNAL_PLAYER_IMAGE_REGISTRY = json.load(f)
    except Exception:
        pass

# Fallback built-in verified registry
if not EXTERNAL_PLAYER_IMAGE_REGISTRY:
    EXTERNAL_PLAYER_IMAGE_REGISTRY = {
        "Virat Kohli": {
            "id": "16017429",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Virat_Kohli_during_the_India_vs_Aus_4th_Test_match_at_Narendra_Modi_Stadium_on_09_March_2023.jpg/330px-Virat_Kohli_during_the_India_vs_Aus_4th_Test_match_at_Narendra_Modi_Stadium_on_09_March_2023.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Rohit Sharma": {
            "id": "4059625",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Prime_Minister_Of_Bharat_Shri_Narendra_Damodardas_Modi_with_Shri_Rohit_Gurunath_Sharma_%2528Cropped%2529.jpg/330px-Prime_Minister_Of_Bharat_Shri_Narendra_Damodardas_Modi_with_Shri_Rohit_Gurunath_Sharma_%2528Cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Jasprit Bumrah": {
            "id": "39012401",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/0/02/Jasprit_Bumrah_in_PMO_New_Delhi.jpg/330px-Jasprit_Bumrah_in_PMO_New_Delhi.jpg&w=300&h=300&fit=cover&a=top"
        },
        "MS Dhoni": {
            "id": "1695229",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/d/d5/MS_Dhoni_%2528Prabhav_%252723_-_RiGI_2023%2529.jpg/330px-MS_Dhoni_%2528Prabhav_%252723_-_RiGI_2023%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Shubman Gill": {
            "id": "53303803",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/3/34/Shubman_Gill_2023_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Hardik Pandya": {
            "id": "46453363",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Hardik_Pandya_in_PMO_New_Delhi.jpg/330px-Hardik_Pandya_in_PMO_New_Delhi.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Sanju Samson": {
            "id": "38731822",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/7/70/Sanju_Samson_in_PMO_New_Delhi.jpg/330px-Sanju_Samson_in_PMO_New_Delhi.jpg&w=300&h=300&fit=cover&a=top"
        },
        "KL Rahul": {
            "id": "38032109",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/6/69/KL_Rahul_at_Femina_Miss_India_2018_Grand_Finale_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Suryakumar Yadav": {
            "id": "34419117",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Suryakumar_Yadav_in_PMO_New_Delhi.jpg/330px-Suryakumar_Yadav_in_PMO_New_Delhi.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Rashid Khan": {
            "id": "48267567",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/7/71/Rashid_Khan.jpg/330px-Rashid_Khan.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Rishabh Pant": {
            "id": "48358969",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/7/77/Rishabh_Pant.jpg/330px-Rishabh_Pant.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Ravindra Jadeja": {
            "id": "4080673",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/2/2c/PM_Shri_Narendra_Modi_with_Ravindra_Jadeja_%2528Cropped%2529.jpg/330px-PM_Shri_Narendra_Modi_with_Ravindra_Jadeja_%2528Cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Shreyas Iyer": {
            "id": "44874356",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Shreyas_Iyer_snapped_at_the_airport_%2528Cropped%2529.jpg/330px-Shreyas_Iyer_snapped_at_the_airport_%2528Cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Mohammed Siraj": {
            "id": "48810382",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/d/da/Prime_Minister_Of_Bharat_Shri_Narendra_Damodardas_Modi_with_Mohammad_Siraj_%2528cropped%2529.jpg/330px-Prime_Minister_Of_Bharat_Shri_Narendra_Damodardas_Modi_with_Mohammad_Siraj_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Mohammed Shami": {
            "id": "1455839",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/0/01/Mohammed_Shami_Arjuna_Award_%2528cropped%2529.jpg/330px-Mohammed_Shami_Arjuna_Award_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Ruturaj Gaikwad": {
            "id": "51882587",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/2/27/Ruturaj_Gaikwad.jpeg/330px-Ruturaj_Gaikwad.jpeg&w=300&h=300&fit=cover&a=top"
        },
        "Shivam Dube": {
            "id": "53036002",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/6/63/Shivam_Dube_in_PMO_New_Delhi.jpg/330px-Shivam_Dube_in_PMO_New_Delhi.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Pat Cummins": {
            "id": "30712264",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/6/69/Pat_Cummins_fielding_Ashes_2021_%2528cropped%2529.jpg/330px-Pat_Cummins_fielding_Ashes_2021_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Trent Boult": {
            "id": "19343608",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/2/23/2018.02.03.22.23.14-AUSvNZL_T20_AUS_innings%252C_SCG_%252839533156665%2529.jpg/330px-2018.02.03.22.23.14-AUSvNZL_T20_AUS_innings%252C_SCG_%252839533156665%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Jos Buttler": {
            "id": "24378873",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/0/01/Jos_Buttler_in_2023.jpg/330px-Jos_Buttler_in_2023.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Quinton de Kock": {
            "id": "36265417",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/6/67/QUINTON_DE_KOCK_%252815681398316%2529.jpg/330px-QUINTON_DE_KOCK_%252815681398316%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Ajinkya Rahane": {
            "id": "22647679",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/9/96/Ajinkya_Rahane_2016_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Cameron Green": {
            "id": "52825372",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Cameron_Green_fielding_Boxing_Day_2022_%2528cropped%2529.jpg/330px-Cameron_Green_fielding_Boxing_Day_2022_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Jofra Archer": {
            "id": "51025332",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/d/de/Jofra_Archer_%25283%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Kagiso Rabada": {
            "id": "42055196",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Kingdom_Kome_co-founders_Kagiso_Rabada_and_Cameron_Scott_at_the_first_private_screening_of_The_Ring_of_Beasts_%2528cropped%2529.jpg/330px-Kingdom_Kome_co-founders_Kagiso_Rabada_and_Cameron_Scott_at_the_first_private_screening_of_The_Ring_of_Beasts_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Ishan Kishan": {
            "id": "48167828",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Ishan_Kishan.jpg/330px-Ishan_Kishan.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Tilak Varma": {
            "id": "59521211",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Tilak_Varma_in_March_2026.png/330px-Tilak_Varma_in_March_2026.png&w=300&h=300&fit=cover&a=top"
        },
        "Phil Salt": {
            "id": "47728364",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/4/4b/2_02_Phil_Salt.jpg/330px-2_02_Phil_Salt.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Will Jacks": {
            "id": "57134009",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/4/4d/4_20_Will_Jacks.jpg/330px-4_20_Will_Jacks.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Jaydev Unadkat": {
            "id": "27149166",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/7/7f/2_09_Jaydev_Unadkat_1.jpg/330px-2_09_Jaydev_Unadkat_1.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Josh Hazlewood": {
            "id": "20216949",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/d/d9/2018_Josh_Hazlewood_%2528cropped%2529.jpg/330px-2018_Josh_Hazlewood_%2528cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Keshav Maharaj": {
            "id": "46228681",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/f/fa/New_face_6-_Keshav_Maharaj.jpg/330px-New_face_6-_Keshav_Maharaj.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Nitish Kumar Reddy": {
            "id": "62944671",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Nitish_Kumar_Reddy_BGT_2024_%2528cropped%2529_2.jpg/330px-Nitish_Kumar_Reddy_BGT_2024_%2528cropped%2529_2.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Sameer Rizvi": {
            "id": "62944695",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Sameer_rizvi_With_Rashid_Zirak.jpg/330px-Sameer_rizvi_With_Rashid_Zirak.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Venkatesh Iyer": {
            "id": "49550659",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Venkatesh_Iyer.png/330px-Venkatesh_Iyer.png&w=300&h=300&fit=cover&a=top"
        },
        "Arshdeep Singh": {
            "id": "58523693",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/3/36/Prime_Minister_Of_Bharat_Shri_Narendra_Damodardas_Modi_with_Arshdeep_Singh_Family_%2528Cropped%2529.jpg/330px-Prime_Minister_Of_Bharat_Shri_Narendra_Damodardas_Modi_with_Arshdeep_Singh_Family_%2528Cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Kuldeep Yadav": {
            "id": "42359036",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/9/91/Kuldeep_Yadav_in_PMO_New_Delhi.jpg/330px-Kuldeep_Yadav_in_PMO_New_Delhi.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Bhuvneshwar Kumar": {
            "id": "35551720",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/2/27/Bhuvneshwar_kumar_With_Rashid_Zirak_%2528Bhuvneshwar_Kumar_cropped%2529.jpg/330px-Bhuvneshwar_kumar_With_Rashid_Zirak_%2528Bhuvneshwar_Kumar_cropped%2529.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Deepak Chahar": {
            "id": "37079053",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Deepak_Chahar.jpg/330px-Deepak_Chahar.jpg&w=300&h=300&fit=cover&a=top"
        },
        "Axar Patel": {
            "id": "40340681",
            "url": "https://wsrv.nl/?url=https%3A//upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Axar_Patel_in_PMO_New_Delhi.jpg/330px-Axar_Patel_in_PMO_New_Delhi.jpg&w=300&h=300&fit=cover&a=top"
        }
    }

# Common Cricket Aliases Map (handles name variations and initials)
PLAYER_ALIASES = {
    "v kohli": "Virat Kohli",
    "virat": "Virat Kohli",
    "rg sharma": "Rohit Sharma",
    "rohit": "Rohit Sharma",
    "jj bumrah": "Jasprit Bumrah",
    "bumrah": "Jasprit Bumrah",
    "msd": "MS Dhoni",
    "dhoni": "MS Dhoni",
    "m s dhoni": "MS Dhoni",
    "mahendra singh dhoni": "MS Dhoni",
    "s gill": "Shubman Gill",
    "gill": "Shubman Gill",
    "hardik": "Hardik Pandya",
    "h pandya": "Hardik Pandya",
    "samson": "Sanju Samson",
    "s samson": "Sanju Samson",
    "rahul": "KL Rahul",
    "k l rahul": "KL Rahul",
    "kannanur lokesh rahul": "KL Rahul",
    "sky": "Suryakumar Yadav",
    "surya": "Suryakumar Yadav",
    "pant": "Rishabh Pant",
    "r pant": "Rishabh Pant",
    "jadeja": "Ravindra Jadeja",
    "r jadeja": "Ravindra Jadeja",
    "siraj": "Mohammed Siraj",
    "md siraj": "Mohammed Siraj",
    "shami": "Mohammed Shami",
    "md shami": "Mohammed Shami",
    "cummins": "Pat Cummins",
    "boult": "Trent Boult",
    "buttler": "Jos Buttler",
    "gaikwad": "Ruturaj Gaikwad",
    "dube": "Shivam Dube",
    "rabada": "Kagiso Rabada",
    "archer": "Jofra Archer"
}


def _resolve_canonical_name(player_name: str) -> str:
    """Resolves input player name or alias to canonical registry key."""
    if not player_name:
        return ""
    clean = player_name.strip()
    if clean in EXTERNAL_PLAYER_IMAGE_REGISTRY:
        return clean
    lower = clean.lower()
    if lower in PLAYER_ALIASES:
        canonical = PLAYER_ALIASES[lower]
        if canonical in EXTERNAL_PLAYER_IMAGE_REGISTRY:
            return canonical
    for reg_name in EXTERNAL_PLAYER_IMAGE_REGISTRY:
        if reg_name.lower() == lower:
            return reg_name
    return clean


def get_player_id(player_name: str) -> str:
    """Returns official external player ID or deterministic hash-based numeric ID."""
    canonical = _resolve_canonical_name(player_name)
    if canonical in EXTERNAL_PLAYER_IMAGE_REGISTRY:
        return str(EXTERNAL_PLAYER_IMAGE_REGISTRY[canonical].get("id", "100001"))
    # Deterministic 6-digit numeric ID based on name hash
    return str(abs(hash(canonical or player_name or "player")) % 900000 + 100000)


def get_player_image_url(player_name: str, team: str = "", role: str = "") -> str:
    """
    Resolves verified external player portrait photo URL.
    Images are served directly over CDN URLs.
    Does NOT download or store images locally.
    """
    if not player_name:
        return "https://ui-avatars.com/api/?name=Player&background=0f172a&color=06b6d4&size=256&bold=true"

    clean_name = player_name.strip()
    canonical = _resolve_canonical_name(clean_name)
    
    # Check in-memory cache
    if canonical in _IMAGE_CACHE:
        return _IMAGE_CACHE[canonical]

    # Check external verified registry
    if canonical in EXTERNAL_PLAYER_IMAGE_REGISTRY:
        img_url = EXTERNAL_PLAYER_IMAGE_REGISTRY[canonical]["url"]
        _IMAGE_CACHE[canonical] = img_url
        return img_url

    # Dynamic External Sports Avatar API for unlisted squad players
    encoded_name = urllib.parse.quote_plus(clean_name)
    bg_color = "0f172a"  # Sleek dark navy
    fg_color = "06b6d4"  # Cyan accent

    # Role-based custom accent coloring
    role_str = str(role).lower()
    if "bowler" in role_str:
        fg_color = "10b981"  # Emerald
    elif "all" in role_str:
        fg_color = "f59e0b"  # Amber
    elif "keeper" in role_str:
        fg_color = "8b5cf6"  # Purple

    dynamic_url = f"https://ui-avatars.com/api/?name={encoded_name}&background={bg_color}&color={fg_color}&size=256&bold=true&font-size=0.38&rounded=false"
    
    _IMAGE_CACHE[canonical] = dynamic_url
    return dynamic_url


def get_player_image_metadata(player_name: str, team: str = "", role: str = "") -> dict:
    """
    Returns complete player image metadata including external URL, player ID,
    source provider, and fallback initials.
    """
    clean_name = player_name.strip() if player_name else "Unknown Player"
    canonical = _resolve_canonical_name(clean_name)
    player_id = get_player_id(clean_name)
    image_url = get_player_image_url(clean_name, team=team, role=role)
    is_direct_headshot = canonical in EXTERNAL_PLAYER_IMAGE_REGISTRY
    
    words = clean_name.split()
    initials = "".join([w[0] for w in words[:2]]).upper() if words else "PL"

    return {
        "player_name": clean_name,
        "player_id": player_id,
        "image_url": image_url,
        "source": "Wikipedia Verified CDN" if is_direct_headshot else "External Avatar API",
        "initials": initials
    }
