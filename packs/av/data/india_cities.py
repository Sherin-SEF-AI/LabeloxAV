"""City to state and urban class, for the AV pack's region model.

Classification follows the Census of India 2011 urban agglomeration definitions, which are the categories
Indian transport planning is actually written in:

  megacity      population 10 million or more
  million_plus  1 million or more, the census "million-plus UA/city" set
  class_1       100,000 or more, the census Class I town definition
  other         everything smaller

Source: Census of India 2011, Provisional Population Totals, Urban Agglomerations and Cities (Table 3,
"Cities having population 1 lakh and above"). The classification is the census's; the mapping of a corpus
city string onto it is ours.

Keys are lowercase and punctuation-free. Aliases matter more than the canonical names here: the corpus
records Bengaluru as `BLR` on 372 sessions and `Bengaluru` on one, which is the same city recorded twice and
the reason a region model exists at all.

Non-Indian cities are deliberately absent rather than given a made-up class. A KITTI drive through Karlsruhe
is not a class_1 town and pretending otherwise puts German motorway footage in an Indian urban stratum.
"""

from __future__ import annotations

from typing import NamedTuple


class City(NamedTuple):
    name: str
    state: str
    urban_class: str


# Canonical records. One entry per city; the alias table below points many strings at each.
CITIES: dict[str, City] = {
    "mumbai": City("Mumbai", "Maharashtra", "megacity"),
    "delhi": City("Delhi", "Delhi", "megacity"),
    "kolkata": City("Kolkata", "West Bengal", "megacity"),
    "bengaluru": City("Bengaluru", "Karnataka", "million_plus"),
    "chennai": City("Chennai", "Tamil Nadu", "million_plus"),
    "hyderabad": City("Hyderabad", "Telangana", "million_plus"),
    "ahmedabad": City("Ahmedabad", "Gujarat", "million_plus"),
    "pune": City("Pune", "Maharashtra", "million_plus"),
    "surat": City("Surat", "Gujarat", "million_plus"),
    "jaipur": City("Jaipur", "Rajasthan", "million_plus"),
    "lucknow": City("Lucknow", "Uttar Pradesh", "million_plus"),
    "kanpur": City("Kanpur", "Uttar Pradesh", "million_plus"),
    "nagpur": City("Nagpur", "Maharashtra", "million_plus"),
    "indore": City("Indore", "Madhya Pradesh", "million_plus"),
    "thane": City("Thane", "Maharashtra", "million_plus"),
    "bhopal": City("Bhopal", "Madhya Pradesh", "million_plus"),
    "visakhapatnam": City("Visakhapatnam", "Andhra Pradesh", "million_plus"),
    "patna": City("Patna", "Bihar", "million_plus"),
    "vadodara": City("Vadodara", "Gujarat", "million_plus"),
    "ghaziabad": City("Ghaziabad", "Uttar Pradesh", "million_plus"),
    "ludhiana": City("Ludhiana", "Punjab", "million_plus"),
    "agra": City("Agra", "Uttar Pradesh", "million_plus"),
    "nashik": City("Nashik", "Maharashtra", "million_plus"),
    "faridabad": City("Faridabad", "Haryana", "million_plus"),
    "meerut": City("Meerut", "Uttar Pradesh", "million_plus"),
    "rajkot": City("Rajkot", "Gujarat", "million_plus"),
    "varanasi": City("Varanasi", "Uttar Pradesh", "million_plus"),
    "srinagar": City("Srinagar", "Jammu and Kashmir", "million_plus"),
    "aurangabad": City("Aurangabad", "Maharashtra", "million_plus"),
    "dhanbad": City("Dhanbad", "Jharkhand", "million_plus"),
    "amritsar": City("Amritsar", "Punjab", "million_plus"),
    "allahabad": City("Prayagraj", "Uttar Pradesh", "million_plus"),
    "ranchi": City("Ranchi", "Jharkhand", "million_plus"),
    "howrah": City("Howrah", "West Bengal", "million_plus"),
    "coimbatore": City("Coimbatore", "Tamil Nadu", "million_plus"),
    "jabalpur": City("Jabalpur", "Madhya Pradesh", "million_plus"),
    "gwalior": City("Gwalior", "Madhya Pradesh", "million_plus"),
    "vijayawada": City("Vijayawada", "Andhra Pradesh", "million_plus"),
    "jodhpur": City("Jodhpur", "Rajasthan", "million_plus"),
    "madurai": City("Madurai", "Tamil Nadu", "million_plus"),
    "raipur": City("Raipur", "Chhattisgarh", "million_plus"),
    "kota": City("Kota", "Rajasthan", "million_plus"),
    "chandigarh": City("Chandigarh", "Chandigarh", "million_plus"),
    "guwahati": City("Guwahati", "Assam", "million_plus"),
    "solapur": City("Solapur", "Maharashtra", "million_plus"),
    "hubli": City("Hubli-Dharwad", "Karnataka", "million_plus"),
    "mysuru": City("Mysuru", "Karnataka", "million_plus"),
    "tiruchirappalli": City("Tiruchirappalli", "Tamil Nadu", "million_plus"),
    "bareilly": City("Bareilly", "Uttar Pradesh", "million_plus"),
    "moradabad": City("Moradabad", "Uttar Pradesh", "million_plus"),
    "gurugram": City("Gurugram", "Haryana", "million_plus"),
    "noida": City("Noida", "Uttar Pradesh", "class_1"),
    "kochi": City("Kochi", "Kerala", "million_plus"),
    "thiruvananthapuram": City("Thiruvananthapuram", "Kerala", "million_plus"),
    "kozhikode": City("Kozhikode", "Kerala", "million_plus"),
    "bhubaneswar": City("Bhubaneswar", "Odisha", "class_1"),
    "dehradun": City("Dehradun", "Uttarakhand", "class_1"),
    "mangaluru": City("Mangaluru", "Karnataka", "class_1"),
    "belagavi": City("Belagavi", "Karnataka", "class_1"),
    "shivamogga": City("Shivamogga", "Karnataka", "class_1"),
    "udupi": City("Udupi", "Karnataka", "other"),
    "goa": City("Panaji", "Goa", "other"),
    "shimla": City("Shimla", "Himachal Pradesh", "other"),
    "puducherry": City("Puducherry", "Puducherry", "class_1"),
}

# Every string a corpus session might carry, pointing at a canonical key. IATA codes are here because that is
# what the fleet actually records: 372 of 377 sessions say `BLR`.
ALIASES: dict[str, str] = {
    # Bengaluru, recorded two ways in the same corpus. This one line is the whole reason for this module.
    "blr": "bengaluru", "bangalore": "bengaluru", "bengalooru": "bengaluru", "bglr": "bengaluru",
    "bom": "mumbai", "bombay": "mumbai",
    "del": "delhi", "new delhi": "delhi", "newdelhi": "delhi", "ncr": "delhi",
    "ccu": "kolkata", "calcutta": "kolkata",
    "maa": "chennai", "madras": "chennai",
    "hyd": "hyderabad", "secunderabad": "hyderabad",
    "amd": "ahmedabad",
    "pnq": "pune", "poona": "pune",
    "jai": "jaipur",
    "lko": "lucknow",
    "ixc": "chandigarh",
    "cok": "kochi", "cochin": "kochi", "ernakulam": "kochi",
    "trv": "thiruvananthapuram", "trivandrum": "thiruvananthapuram",
    "ccj": "kozhikode", "calicut": "kozhikode",
    "gau": "guwahati",
    "bbi": "bhubaneswar",
    "vtz": "visakhapatnam", "vizag": "visakhapatnam",
    "cjb": "coimbatore",
    "ixe": "mangaluru", "mangalore": "mangaluru",
    "mysore": "mysuru",
    "gurgaon": "gurugram",
    "prayagraj": "allahabad",
    "benares": "varanasi", "banaras": "varanasi",
    "trichy": "tiruchirappalli",
    "hubballi": "hubli", "dharwad": "hubli",
    "belgaum": "belagavi",
    "shimoga": "shivamogga",
    "pondicherry": "puducherry",
}

# Strings in the corpus that name a place outside this pack's region model, kept so the resolver can say
# "outside India" rather than "unknown". `GLOBAL` is the Mapillary import, which is not one place at all.
NON_INDIA: dict[str, str] = {
    "berkeley": "United States",
    "karlsruhe": "Germany",
    "global": "mixed",
}
