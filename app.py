import io
import os
import re
import csv
import cv2
import base64
import numpy as np
import ssl
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
import easyocr

# Bypass SSL verification for downloading EasyOCR models
ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI(title="YOLO License Plate Recognition Web App")

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static directory for JS/CSS assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("best.pt")

# Initialize EasyOCR reader
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'])

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Minimum YOLO detection confidence — skip weak/noisy detections
YOLO_CONF_THRESHOLD = 0.40

# Minimum per-token EasyOCR confidence to accept a word
# Lowered to 0.35 for better recall; two-pass system handles false positives
OCR_TOKEN_CONF_THRESHOLD = 0.35

# Characters allowed in OCR — critical for plate accuracy!
OCR_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Indian State & Union Territory Codes Mapping
STATE_CODES = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "WB": "West Bengal"
}

# Standard Indian plate: 2 letters (state) + 2 digits (district) + 1-3 letters (series) + 4 digits
PLATE_REGEX = re.compile(r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$')

# Bharat (BH) series plate: BH + 4-digit year + 2 digits + 1-2 letters
BH_PLATE_REGEX = re.compile(r'^(BH)(\d{4})(\d{2})([A-Z]{1,2})$')

# ------------------------------------------------------------------
# Comprehensive Indian RTO District Code Database
# Key format: "STNN" (state code + 2-digit district number, zero-padded)
# ------------------------------------------------------------------
RTO_CODES = {
    # ── Karnataka (KA) ──
    "KA01": "Bangalore Central",
    "KA02": "Bangalore West (Rajajinagar)",
    "KA03": "Bangalore East",
    "KA04": "Bangalore South",
    "KA05": "Kolar",
    "KA06": "Tumkur",
    "KA07": "Mysuru (Mysore)",
    "KA08": "Hassan",
    "KA09": "Mandya",
    "KA10": "Chamarajanagar",
    "KA11": "Kodagu (Madikeri)",
    "KA12": "Dakshina Kannada (Mangaluru)",
    "KA13": "Udupi",
    "KA14": "Chikkamagaluru",
    "KA15": "Shivamogga (Shimoga)",
    "KA16": "Haveri",
    "KA17": "Dharwad",
    "KA18": "Gadag",
    "KA19": "Belagavi (Belgaum)",
    "KA20": "Vijayapura (Bijapur)",
    "KA21": "Bagalkot",
    "KA22": "Raichur",
    "KA23": "Koppal",
    "KA24": "Ballari (Bellary)",
    "KA25": "Chitradurga",
    "KA26": "Davanagere",
    "KA27": "Bangalore North",
    "KA28": "Bangalore Rural (Devanahalli)",
    "KA29": "Ramanagara",
    "KA30": "Yadgir",
    "KA31": "Chamarajanagar (sub)",
    "KA32": "Bangalore (Rajarajeshwarinagar)",
    "KA33": "Bidar",
    "KA34": "Kalaburagi (Gulbarga)",
    "KA35": "Bangalore (Whitefield)",
    "KA36": "Bangalore (Hebbal)",
    "KA37": "Bangalore (Indiranagar)",
    "KA38": "Bangalore (JP Nagar)",
    "KA39": "Bangalore (Sarjapur)",
    "KA40": "Bangalore (Marathahalli)",
    "KA41": "Vijayanagara (Hosapete)",
    "KA42": "Chikkaballapur",
    "KA43": "Bangalore (Koramangala)",
    "KA44": "Bangalore (Yelahanka)",
    "KA45": "Bangalore (Jayanagar)",
    "KA46": "Bangalore (Malleswaram)",
    "KA47": "Bangalore (Shivajinagar)",
    "KA48": "Bangalore (Basavanagudi)",
    "KA49": "Bangalore (KR Puram)",
    "KA50": "Bangalore (Electronic City)",
    "KA51": "Vijayanagara (District)",
    "KA52": "Mysuru (Sub)",
    "KA53": "Belagavi (Sub)",
    "KA54": "Davanagere (Sub)",
    "KA55": "Tumkur (Sub)",

    # ── Maharashtra (MH) ──
    "MH01": "Mumbai (South)",
    "MH02": "Mumbai (West)",
    "MH03": "Mumbai (East)",
    "MH04": "Thane",
    "MH05": "Kalyan",
    "MH06": "Raigad (Alibag)",
    "MH07": "Pune (City)",
    "MH08": "Nashik",
    "MH09": "Ahmednagar",
    "MH10": "Dhule",
    "MH11": "Jalgaon",
    "MH12": "Aurangabad",
    "MH13": "Latur",
    "MH14": "Osmanabad",
    "MH15": "Solapur",
    "MH16": "Kolhapur",
    "MH17": "Sangli",
    "MH18": "Satara",
    "MH19": "Ratnagiri",
    "MH20": "Sindhudurg",
    "MH21": "Amravati",
    "MH22": "Akola",
    "MH23": "Buldhana",
    "MH24": "Washim",
    "MH25": "Yavatmal",
    "MH26": "Wardha",
    "MH27": "Nagpur",
    "MH28": "Bhandara",
    "MH29": "Chandrapur",
    "MH30": "Gadchiroli",
    "MH31": "Gondia",
    "MH32": "Nandurbar",
    "MH33": "Navi Mumbai",
    "MH34": "Mira-Bhayandar",
    "MH35": "Vasai-Virar",
    "MH36": "Nashik (West)",
    "MH37": "Malegaon",
    "MH38": "Pimpri-Chinchwad",
    "MH39": "Solapur (West)",
    "MH40": "Jalna",
    "MH41": "Parbhani",
    "MH42": "Hingoli",
    "MH43": "Nanded",
    "MH44": "Beed",
    "MH45": "Ulhasnagar",
    "MH46": "Bhiwandi",
    "MH47": "Alibag",
    "MH48": "Pune (Rural)",
    "MH49": "Baramati",
    "MH50": "Pandharpur",

    # ── Delhi (DL) ──
    "DL01": "Delhi (North)",
    "DL02": "Delhi (South)",
    "DL03": "Delhi (East)",
    "DL04": "Delhi (West)",
    "DL05": "Delhi (Central)",
    "DL06": "Delhi (Loni)",
    "DL07": "Delhi (Rohini)",
    "DL08": "Delhi (Saket)",
    "DL09": "Delhi (Janakpuri)",
    "DL10": "Delhi (Dwarka)",
    "DL11": "Delhi (Vasant Vihar)",
    "DL12": "Delhi (Mayur Vihar)",
    "DL13": "Delhi (Shahdara)",
    "DL14": "Delhi (Badarpur)",

    # ── Tamil Nadu (TN) ──
    "TN01": "Chennai (Central)",
    "TN02": "Chennai (South)",
    "TN03": "Chennai (West)",
    "TN04": "Chennai (North)",
    "TN05": "Chennai (East)",
    "TN06": "Chennai (Southeast)",
    "TN07": "Tiruvallur",
    "TN08": "Kanchipuram",
    "TN09": "Vellore",
    "TN10": "Tiruvannamalai",
    "TN11": "Villupuram",
    "TN12": "Cuddalore",
    "TN13": "Pudukottai",
    "TN14": "Thanjavur",
    "TN15": "Nagapattinam",
    "TN16": "Tiruvarur",
    "TN17": "Tiruchirappalli",
    "TN18": "Perambalur",
    "TN19": "Ariyalur",
    "TN20": "Salem",
    "TN21": "Namakkal",
    "TN22": "Dharmapuri",
    "TN23": "Krishnagiri",
    "TN24": "Erode",
    "TN25": "Coimbatore (North)",
    "TN26": "Coimbatore (South)",
    "TN27": "Nilgiris (Ooty)",
    "TN28": "Tiruppur",
    "TN29": "Dindigul",
    "TN30": "Madurai (North)",
    "TN31": "Madurai (South)",
    "TN32": "Theni",
    "TN33": "Virudhunagar",
    "TN34": "Ramanathapuram",
    "TN35": "Sivaganga",
    "TN36": "Tirunelveli (North)",
    "TN37": "Tirunelveli (South)",
    "TN38": "Thoothukudi",
    "TN39": "Kanyakumari",
    "TN40": "Chennai (Avadi)",
    "TN41": "Chengalpattu",
    "TN42": "Ranipet",
    "TN43": "Tirupattur",
    "TN44": "Kallakurichi",
    "TN45": "Tenkasi",
    "TN46": "Nagercoil",

    # ── Telangana (TS) ──
    "TS01": "Hyderabad (West)",
    "TS02": "Hyderabad (North)",
    "TS03": "Hyderabad (East)",
    "TS04": "Hyderabad (South)",
    "TS05": "Ranga Reddy",
    "TS06": "Medak",
    "TS07": "Nizamabad",
    "TS08": "Adilabad",
    "TS09": "Karimnagar",
    "TS10": "Warangal",
    "TS11": "Khammam",
    "TS12": "Nalgonda",
    "TS13": "Mahabubnagar",
    "TS14": "Hyderabad (Central)",
    "TS15": "Cyberabad (Madhapur)",
    "TS16": "Medchal",
    "TS17": "Sangareddy",
    "TS18": "Vikarabad",
    "TS19": "Siddipet",
    "TS20": "Yadadri Bhuvanagiri",
    "TS21": "Suryapet",
    "TS22": "Jangaon",
    "TS23": "Mulugu",
    "TS24": "Bhadradri Kothagudem",
    "TS25": "Mancherial",
    "TS26": "Peddapalli",
    "TS27": "Rajanna Sircilla",
    "TS28": "Kamareddy",
    "TS29": "Nirmal",
    "TS30": "Jagitial",
    "TS31": "Kumuram Bheem Asifabad",
    "TS32": "Nagarkurnool",
    "TS33": "Wanaparthy",
    "TS34": "Narayanpet",
    "TS35": "Gadwal",

    # ── Andhra Pradesh (AP) ──
    "AP01": "Kurnool",
    "AP02": "Anantapur",
    "AP03": "Chittoor (North)",
    "AP04": "YSR Kadapa",
    "AP05": "Nellore",
    "AP06": "Guntur",
    "AP07": "Krishna (Machilipatnam)",
    "AP08": "West Godavari (Eluru)",
    "AP09": "East Godavari (Kakinada)",
    "AP10": "Visakhapatnam (North)",
    "AP11": "Vizianagaram",
    "AP12": "Srikakulam",
    "AP13": "Vijayawada",
    "AP14": "Tirupati",
    "AP15": "Chittoor (South)",
    "AP16": "Prakasam (Ongole)",
    "AP17": "Rajampet",
    "AP18": "Rajahmundry",
    "AP19": "Eluru",
    "AP20": "Visakhapatnam (South)",
    "AP21": "Nandyal",
    "AP22": "Proddatur",
    "AP23": "Visakhapatnam (City)",
    "AP24": "Bhimavaram",
    "AP25": "Srikakulam (Sub)",
    "AP26": "Gudivada",
    "AP27": "Tenali",
    "AP28": "Narsaraopeta",
    "AP29": "Sri Sathya Sai (Puttaparthi)",
    "AP30": "Anakapalli",
    "AP31": "Parvathipuram Manyam",
    "AP32": "Alluri Sitharama Raju",
    "AP33": "Konaseema (Amalapuram)",
    "AP34": "Eluru (New)",
    "AP35": "Bapatla",

    # ── Gujarat (GJ) ──
    "GJ01": "Ahmedabad (South)",
    "GJ02": "Ahmedabad (North)",
    "GJ03": "Gandhinagar",
    "GJ04": "Mehsana",
    "GJ05": "Sabarkantha (Himmatnagar)",
    "GJ06": "Banaskantha (Palanpur)",
    "GJ07": "Patan",
    "GJ08": "Vadodara (South)",
    "GJ09": "Vadodara (North)",
    "GJ10": "Anand",
    "GJ11": "Kheda (Nadiad)",
    "GJ12": "Bharuch",
    "GJ13": "Narmada (Rajpipla)",
    "GJ14": "Surat (South)",
    "GJ15": "Surat (North)",
    "GJ16": "Navsari",
    "GJ17": "Valsad",
    "GJ18": "Dang (Ahwa)",
    "GJ19": "Rajkot (South)",
    "GJ20": "Rajkot (North)",
    "GJ21": "Surendranagar",
    "GJ22": "Jamnagar (South)",
    "GJ23": "Jamnagar (North)",
    "GJ24": "Junagadh (South)",
    "GJ25": "Junagadh (North)",
    "GJ26": "Amreli",
    "GJ27": "Bhavnagar (South)",
    "GJ28": "Bhavnagar (North)",
    "GJ29": "Botad",
    "GJ30": "Morbi",
    "GJ31": "Kutch (Bhuj)",
    "GJ32": "Panchmahal (Godhra)",
    "GJ33": "Dahod",
    "GJ34": "Chhota Udaipur",
    "GJ35": "Aravalli (Modasa)",
    "GJ36": "Mahisagar (Lunawada)",
    "GJ37": "Ahmedabad (East)",
    "GJ38": "Surat (East)",
    "GJ39": "Vadodara (East)",
    "GJ40": "Devbhumi Dwarka",
    "GJ41": "Gir Somnath",
    "GJ42": "Porbandar",

    # ── Rajasthan (RJ) ──
    "RJ01": "Ajmer (North)",
    "RJ02": "Ajmer (South)",
    "RJ03": "Alwar",
    "RJ04": "Banswara",
    "RJ05": "Baran",
    "RJ06": "Barmer",
    "RJ07": "Bharatpur",
    "RJ08": "Bhilwara",
    "RJ09": "Bikaner",
    "RJ10": "Bundi",
    "RJ11": "Chittorgarh",
    "RJ12": "Churu",
    "RJ13": "Dausa",
    "RJ14": "Dholpur",
    "RJ15": "Dungarpur",
    "RJ16": "Hanumangarh",
    "RJ17": "Jaipur (North)",
    "RJ18": "Jaipur (South)",
    "RJ19": "Jaisalmer",
    "RJ20": "Jalore",
    "RJ21": "Jhalawar",
    "RJ22": "Jhunjhunu",
    "RJ23": "Jodhpur (North)",
    "RJ24": "Jodhpur (South)",
    "RJ25": "Karauli",
    "RJ26": "Kota (North)",
    "RJ27": "Kota (South)",
    "RJ28": "Nagaur",
    "RJ29": "Pali",
    "RJ30": "Pratapgarh",
    "RJ31": "Rajsamand",
    "RJ32": "Sawai Madhopur",
    "RJ33": "Sikar",
    "RJ34": "Sirohi",
    "RJ35": "Sri Ganganagar",
    "RJ36": "Tonk",
    "RJ37": "Udaipur",
    "RJ38": "Jaipur (East)",
    "RJ39": "Jodhpur (Rural)",
    "RJ40": "Kota (Rural)",
    "RJ41": "Bikaner (Rural)",
    "RJ42": "Alwar (Rural)",
    "RJ43": "Bharatpur (Rural)",
    "RJ44": "Udaipur (Rural)",
    "RJ45": "Jodhpur (West)",
    "RJ46": "Jaipur (West)",
    "RJ47": "Jaipur (Rural)",
    "RJ48": "Udaipur (Sub)",

    # ── Uttar Pradesh (UP) ──
    "UP11": "Agra",
    "UP12": "Firozabad",
    "UP13": "Mainpuri",
    "UP14": "Mathura",
    "UP15": "Aligarh",
    "UP16": "Etah",
    "UP17": "Hathras",
    "UP18": "Kasganj",
    "UP19": "Bulandshahr",
    "UP20": "Gautam Buddh Nagar (Noida)",
    "UP21": "Bareilly",
    "UP22": "Badaun",
    "UP23": "Pilibhit",
    "UP24": "Shahjahanpur",
    "UP25": "Bijnor",
    "UP26": "Amroha (J P Nagar)",
    "UP27": "Moradabad",
    "UP28": "Rampur",
    "UP29": "Saharanpur",
    "UP30": "Muzaffarnagar",
    "UP31": "Shamli",
    "UP32": "Meerut",
    "UP33": "Hapur",
    "UP34": "Ghaziabad",
    "UP35": "Agra (West)",
    "UP36": "Auraiya",
    "UP37": "Kannauj",
    "UP38": "Etawah",
    "UP39": "Farrukhabad",
    "UP40": "Kanpur (North)",
    "UP41": "Kanpur (South)",
    "UP42": "Kanpur Dehat",
    "UP43": "Hamirpur",
    "UP44": "Jalaun (Orai)",
    "UP45": "Jhansi",
    "UP46": "Lalitpur",
    "UP47": "Mahoba",
    "UP48": "Banda",
    "UP49": "Chitrakoot",
    "UP50": "Fatehpur",
    "UP51": "Kaushambi",
    "UP52": "Prayagraj (Allahabad)",
    "UP53": "Lucknow (East)",
    "UP54": "Lucknow (West)",
    "UP55": "Hardoi",
    "UP56": "Lakhimpur Kheri",
    "UP57": "Raebareli",
    "UP58": "Sitapur",
    "UP59": "Unnao",
    "UP60": "Ayodhya (Faizabad)",
    "UP61": "Ambedkar Nagar",
    "UP62": "Amethi",
    "UP63": "Barabanki",
    "UP64": "Sultanpur",
    "UP65": "Azamgarh",
    "UP66": "Ballia",
    "UP67": "Mau",
    "UP68": "Jaunpur",
    "UP69": "Bhadohi",
    "UP70": "Ghazipur",
    "UP71": "Varanasi (North)",
    "UP72": "Varanasi (South)",
    "UP73": "Chandauli",
    "UP74": "Mirzapur",
    "UP75": "Sonbhadra",
    "UP76": "Gorakhpur",
    "UP77": "Deoria",
    "UP78": "Kushinagar",
    "UP79": "Maharajganj",
    "UP80": "Basti",
    "UP81": "Siddharthnagar",
    "UP82": "Sant Kabir Nagar",
    "UP83": "Bahraich",
    "UP84": "Balrampur",
    "UP85": "Gonda",
    "UP86": "Shravasti",
    "UP87": "Lucknow (Central)",
    "UP88": "Ghaziabad (Sub)",

    # ── Punjab (PB) ──
    "PB01": "Mohali (SAS Nagar)",
    "PB02": "Ludhiana",
    "PB03": "Jalandhar (North)",
    "PB04": "Jalandhar (South)",
    "PB05": "Amritsar (North)",
    "PB06": "Amritsar (South)",
    "PB07": "Gurdaspur",
    "PB08": "Hoshiarpur",
    "PB09": "Shaheed Bhagat Singh Nagar (Nawanshahr)",
    "PB10": "Kapurthala",
    "PB11": "Moga",
    "PB12": "Firozpur (North)",
    "PB13": "Firozpur (South)",
    "PB14": "Faridkot",
    "PB15": "Muktsar (Sri Muktsar Sahib)",
    "PB16": "Fazilka",
    "PB17": "Ferozepur",
    "PB18": "Bathinda (North)",
    "PB19": "Bathinda (South)",
    "PB20": "Mansa",
    "PB21": "Barnala",
    "PB22": "Sangrur",
    "PB23": "Patiala (North)",
    "PB24": "Patiala (South)",
    "PB25": "Fatehgarh Sahib",
    "PB26": "Rupnagar (Ropar)",
    "PB27": "Tarn Taran",
    "PB28": "Pathankot",
    "PB29": "Malerkotla",
    "PB30": "Ludhiana (Rural)",

    # ── Haryana (HR) ──
    "HR01": "Ambala",
    "HR02": "Panchkula",
    "HR03": "Yamunanagar",
    "HR04": "Kurukshetra",
    "HR05": "Kaithal",
    "HR06": "Karnal",
    "HR07": "Panipat",
    "HR08": "Sonipat",
    "HR09": "Rohtak",
    "HR10": "Jhajjar",
    "HR11": "Faridabad",
    "HR12": "Palwal",
    "HR13": "Nuh (Mewat)",
    "HR14": "Gurugram (Gurgaon)",
    "HR15": "Rewari",
    "HR16": "Mahendragarh (Narnaul)",
    "HR17": "Bhiwani",
    "HR18": "Charkhi Dadri",
    "HR19": "Hisar",
    "HR20": "Fatehabad",
    "HR21": "Sirsa",
    "HR22": "Jind",
    "HR23": "Narnaul",
    "HR24": "Bahadurgarh",
    "HR25": "Hansi",
    "HR26": "Gurugram (DLF / Sohna Road)",

    # ── West Bengal (WB) ──
    "WB01": "Kolkata (North)",
    "WB02": "Kolkata (South)",
    "WB03": "Kolkata (Central)",
    "WB04": "Howrah",
    "WB05": "Hooghly (Chinsurah)",
    "WB06": "North 24 Parganas (Barasat)",
    "WB07": "South 24 Parganas (Alipore)",
    "WB08": "Nadia (Krishnanagar)",
    "WB09": "Murshidabad (Berhampore)",
    "WB10": "Birbhum (Suri)",
    "WB11": "Purba Bardhaman",
    "WB12": "Paschim Bardhaman (Asansol)",
    "WB13": "Bankura",
    "WB14": "Purulia",
    "WB15": "Jhargram",
    "WB16": "Purba Medinipur (Tamluk)",
    "WB17": "Paschim Medinipur (Midnapore)",
    "WB18": "Darjeeling",
    "WB19": "Kalimpong",
    "WB20": "Jalpaiguri",
    "WB21": "Alipurduar",
    "WB22": "Cooch Behar",
    "WB23": "Uttar Dinajpur (Raiganj)",
    "WB24": "Dakshin Dinajpur (Balurghat)",
    "WB25": "Malda",

    # ── Bihar (BR) ──
    "BR01": "Patna (City)",
    "BR02": "Patna (Rural)",
    "BR03": "Nalanda (Bihar Sharif)",
    "BR04": "Gaya",
    "BR05": "Aurangabad",
    "BR06": "Arwal",
    "BR07": "Jehanabad",
    "BR08": "Rohtas (Sasaram)",
    "BR09": "Bhojpur (Ara)",
    "BR10": "Buxar",
    "BR11": "Kaimur (Bhabua)",
    "BR12": "Saran (Chapra)",
    "BR13": "Siwan",
    "BR14": "Gopalganj",
    "BR15": "Muzaffarpur",
    "BR16": "Sitamarhi",
    "BR17": "Sheohar",
    "BR18": "Darbhanga",
    "BR19": "Madhubani",
    "BR20": "Samastipur",
    "BR21": "Begusarai",
    "BR22": "Khagaria",
    "BR23": "Bhagalpur",
    "BR24": "Banka",
    "BR25": "Munger",
    "BR26": "Lakhisarai",
    "BR27": "Sheikhpura",
    "BR28": "Jamui",
    "BR29": "Nawada",
    "BR30": "Gaya (Rural)",
    "BR31": "Supaul",
    "BR32": "Saharsa",
    "BR33": "Madhepura",
    "BR34": "Purnea",
    "BR35": "Araria",
    "BR36": "Kishanganj",
    "BR37": "Katihar",
    "BR38": "West Champaran (Bettiah)",
    "BR39": "East Champaran (Motihari)",
    "BR40": "Vaishali (Hajipur)",

    # ── Madhya Pradesh (MP) ──
    "MP01": "Bhopal",
    "MP02": "Raisen",
    "MP03": "Sehore",
    "MP04": "Vidisha",
    "MP05": "Gwalior",
    "MP06": "Datia",
    "MP07": "Shivpuri",
    "MP08": "Guna",
    "MP09": "Ashoknagar",
    "MP10": "Morena",
    "MP11": "Bhind",
    "MP12": "Sheopur",
    "MP13": "Indore",
    "MP14": "Dhar",
    "MP15": "Jhabua",
    "MP16": "Alirajpur",
    "MP17": "Ujjain",
    "MP18": "Shajapur",
    "MP19": "Agar Malwa",
    "MP20": "Ratlam",
    "MP21": "Mandsaur",
    "MP22": "Neemuch",
    "MP23": "Dewas",
    "MP24": "Jabalpur",
    "MP25": "Katni",
    "MP26": "Narsinghpur",
    "MP27": "Chhindwara",
    "MP28": "Seoni",
    "MP29": "Balaghat",
    "MP30": "Mandla",
    "MP31": "Dindori",
    "MP32": "Umaria",
    "MP33": "Shahdol",
    "MP34": "Anuppur",
    "MP35": "Satna",
    "MP36": "Rewa",
    "MP37": "Sidhi",
    "MP38": "Singrauli",
    "MP39": "Sagar",
    "MP40": "Damoh",
    "MP41": "Chhatarpur",
    "MP42": "Tikamgarh",
    "MP43": "Panna",
    "MP44": "Narmadapuram (Hoshangabad)",
    "MP45": "Betul",
    "MP46": "Harda",
    "MP47": "Khandwa (East Nimar)",
    "MP48": "Burhanpur",
    "MP49": "Khargone (West Nimar)",
    "MP50": "Barwani",
    "MP51": "Indore (Rural)",

    # ── Odisha (OD) ──
    "OD01": "Cuttack",
    "OD02": "Bhubaneswar",
    "OD03": "Puri",
    "OD04": "Nayagarh",
    "OD05": "Khurda",
    "OD06": "Ganjam (Chatrapur)",
    "OD07": "Berhampur",
    "OD08": "Kandhamal (Phulbani)",
    "OD09": "Koraput",
    "OD10": "Nabarangapur",
    "OD11": "Rayagada",
    "OD12": "Kalahandi (Bhawanipatna)",
    "OD13": "Nuapada",
    "OD14": "Bolangir",
    "OD15": "Subarnapur (Sonepur)",
    "OD16": "Sambalpur",
    "OD17": "Bargarh",
    "OD18": "Jharsuguda",
    "OD19": "Sundargarh",
    "OD20": "Rourkela",
    "OD21": "Keonjhar",
    "OD22": "Mayurbhanj (Baripada)",
    "OD23": "Balasore",
    "OD24": "Bhadrak",
    "OD25": "Kendrapara",
    "OD26": "Jagatsinghpur",
    "OD27": "Jajpur",
    "OD28": "Dhenkanal",
    "OD29": "Angul",
    "OD30": "Debagarh",

    # ── Himachal Pradesh (HP) ──
    "HP01": "Shimla",
    "HP02": "Solan",
    "HP03": "Sirmaur (Nahan)",
    "HP04": "Una",
    "HP05": "Hamirpur",
    "HP06": "Bilaspur",
    "HP07": "Mandi",
    "HP08": "Kullu",
    "HP09": "Lahaul & Spiti (Keylong)",
    "HP10": "Kinnaur (Reckong Peo)",
    "HP11": "Chamba",
    "HP12": "Kangra (Dharamsala)",
    "HP13": "Shimla (Rural)",

    # ── Assam (AS) ──
    "AS01": "Kamrup (Guwahati)",
    "AS02": "Kamrup (Rural)",
    "AS03": "Darrang (Mangaldoi)",
    "AS04": "Nagaon",
    "AS05": "Golaghat",
    "AS06": "Jorhat",
    "AS07": "Sivasagar",
    "AS08": "Dibrugarh",
    "AS09": "Tinsukia",
    "AS10": "Lakhimpur (North)",
    "AS11": "Dhemaji",
    "AS12": "Sonitpur (Tezpur)",
    "AS13": "Biswanath",
    "AS14": "Barpeta",
    "AS15": "Nalbari",
    "AS16": "Baksa",
    "AS17": "Chirang (Kajalgaon)",
    "AS18": "Bongaigaon",
    "AS19": "Dhubri",
    "AS20": "South Salmara-Mankachar",
    "AS21": "Goalpara",
    "AS22": "Kamrup (Metro)",
    "AS23": "Karbi Anglong (Diphu)",
    "AS24": "West Karbi Anglong",
    "AS25": "Dima Hasao (Haflong)",
    "AS26": "Cachar (Silchar)",
    "AS27": "Karimganj",
    "AS28": "Hailakandi",
    "AS29": "Majuli",

    # ── Kerala (KL) ──
    "KL01": "Thiruvananthapuram (City)",
    "KL02": "Thiruvananthapuram (District)",
    "KL03": "Kollam",
    "KL04": "Pathanamthitta",
    "KL05": "Alappuzha",
    "KL06": "Kottayam",
    "KL07": "Idukki (Painavu)",
    "KL08": "Ernakulam",
    "KL09": "Thrissur",
    "KL10": "Palakkad",
    "KL11": "Malappuram",
    "KL12": "Kozhikode",
    "KL13": "Wayanad (Kalpetta)",
    "KL14": "Kannur",
    "KL15": "Kasaragod",
    "KL16": "Kochi (City)",
    "KL17": "Thiruvananthapuram (South)",

    # ── Uttarakhand (UK / UA) ──
    "UK01": "Dehradun",
    "UK02": "Haridwar",
    "UK03": "Tehri Garhwal (New Tehri)",
    "UK04": "Uttarkashi",
    "UK05": "Chamoli (Gopeshwar)",
    "UK06": "Rudraprayag",
    "UK07": "Pauri Garhwal",
    "UK08": "Almora",
    "UK09": "Bageshwar",
    "UK10": "Nainital",
    "UK11": "Udham Singh Nagar (Rudrapur)",
    "UK12": "Champawat",
    "UK13": "Pithoragarh",
    "UA01": "Dehradun",
    "UA02": "Haridwar",
    "UA03": "Tehri Garhwal",
    "UA04": "Uttarkashi",
    "UA05": "Chamoli",
    "UA06": "Rudraprayag",
    "UA07": "Pauri Garhwal",
    "UA08": "Almora",
    "UA09": "Bageshwar",
    "UA10": "Nainital",
    "UA11": "Udham Singh Nagar",
    "UA12": "Champawat",
    "UA13": "Pithoragarh",

    # ── Jharkhand (JH) ──
    "JH01": "Ranchi",
    "JH02": "Lohardaga",
    "JH03": "Gumla",
    "JH04": "Simdega",
    "JH05": "Dhanbad",
    "JH06": "Bokaro",
    "JH07": "Hazaribagh",
    "JH08": "Ramgarh",
    "JH09": "Chatra",
    "JH10": "Koderma",
    "JH11": "Giridih",
    "JH12": "Deoghar",
    "JH13": "Godda",
    "JH14": "Dumka",
    "JH15": "Pakur",
    "JH16": "Sahibganj",
    "JH17": "Jamtara",
    "JH18": "Saraikela-Kharsawan",
    "JH19": "West Singhbhum (Chaibasa)",
    "JH20": "East Singhbhum (Jamshedpur)",
    "JH21": "Khunti",
    "JH22": "Latehar",
    "JH23": "Palamu (Daltonganj)",
    "JH24": "Garhwa",

    # ── Chhattisgarh (CG) ──
    "CG01": "Raipur",
    "CG02": "Durg",
    "CG03": "Rajnandgaon",
    "CG04": "Kabirdham (Kawardha)",
    "CG05": "Bilaspur",
    "CG06": "Janjgir-Champa",
    "CG07": "Raigarh",
    "CG08": "Korba",
    "CG09": "Jashpur",
    "CG10": "Surguja (Ambikapur)",
    "CG11": "Korea (Baikunthpur)",
    "CG12": "Surajpur",
    "CG13": "Balrampur",
    "CG14": "Gariaband",
    "CG15": "Balod",
    "CG16": "Bemetara",
    "CG17": "Mungeli",
    "CG18": "Bastar (Jagdalpur)",
    "CG19": "Kondagaon",
    "CG20": "Kanker",
    "CG21": "Narayanpur",
    "CG22": "Bijapur",
    "CG23": "Dantewada",
    "CG24": "Sukma",
    "CG25": "Mahasamund",
    "CG26": "Dhamtari",
    "CG27": "Sarangarh-Bilaigarh",
    "CG28": "Gaurela-Pendra-Marwahi",
    "CG29": "Khairagarh-Chhuikhadan-Gandai",
    "CG30": "Manendragarh-Chirmiri-Bharatpur",

    # ── Goa (GA) ──
    "GA01": "Panaji (North Goa)",
    "GA02": "Margao (South Goa)",
    "GA03": "Mapusa",
    "GA04": "Vasco da Gama",
    "GA05": "Bicholim",
    "GA06": "Ponda",
    "GA07": "Valpoi",
    "GA08": "Quepem",
    "GA09": "Canacona",

    # ── Chandigarh (CH) ──
    "CH01": "Chandigarh (Zone 1)",
    "CH02": "Chandigarh (Zone 2)",
    "CH03": "Chandigarh (Zone 3)",
    "CH04": "Chandigarh (Zone 4)",

    # ── Jammu & Kashmir (JK) ──
    "JK01": "Srinagar",
    "JK02": "Budgam",
    "JK03": "Ganderbal",
    "JK04": "Anantnag",
    "JK05": "Kulgam",
    "JK06": "Shopian",
    "JK07": "Pulwama",
    "JK08": "Baramulla",
    "JK09": "Kupwara",
    "JK10": "Bandipora",
    "JK11": "Jammu",
    "JK12": "Kathua",
    "JK13": "Udhampur",
    "JK14": "Reasi",
    "JK15": "Ramban",
    "JK16": "Doda",
    "JK17": "Kishtwar",
    "JK18": "Samba",
    "JK19": "Rajouri",
    "JK20": "Poonch",

    # ── Ladakh (LA) ──
    "LA01": "Leh",
    "LA02": "Kargil",

    # ── Nagaland (NL) ──
    "NL01": "Kohima",
    "NL02": "Dimapur",
    "NL03": "Mokokchung",
    "NL04": "Tuensang",
    "NL05": "Mon",
    "NL06": "Zunheboto",
    "NL07": "Wokha",
    "NL08": "Phek",
    "NL09": "Longleng",
    "NL10": "Peren",
    "NL11": "Kiphire",
    "NL12": "Noklak",

    # ── Manipur (MN) ──
    "MN01": "Imphal (West)",
    "MN02": "Imphal (East)",
    "MN03": "Bishnupur",
    "MN04": "Thoubal",
    "MN05": "Ukhrul",
    "MN06": "Senapati",
    "MN07": "Tamenglong",
    "MN08": "Churachandpur",
    "MN09": "Chandel",
    "MN10": "Jiribam",
    "MN11": "Kakching",
    "MN12": "Kamjong",
    "MN13": "Noney",
    "MN14": "Pherzawl",
    "MN15": "Tengnoupal",

    # ── Meghalaya (ML) ──
    "ML01": "East Khasi Hills (Shillong)",
    "ML02": "West Khasi Hills (Nongstoin)",
    "ML03": "Ri-Bhoi (Nongpoh)",
    "ML04": "East Jaintia Hills (Khliehriat)",
    "ML05": "West Jaintia Hills (Jowai)",
    "ML06": "East Garo Hills (Tura)",
    "ML07": "West Garo Hills (Dalu)",
    "ML08": "South Garo Hills (Baghmara)",
    "ML09": "Eastern West Khasi Hills",
    "ML10": "South West Garo Hills",
    "ML11": "South West Khasi Hills",
    "ML12": "Eastern West Khasi Hills (Mairang)",

    # ── Tripura (TR) ──
    "TR01": "West Tripura (Agartala)",
    "TR02": "South Tripura (Udaipur)",
    "TR03": "North Tripura (Dharmanagar)",
    "TR04": "Dhalai (Ambassa)",
    "TR05": "Sepahijala (Bishalgarh)",
    "TR06": "Gomati (Udaipur)",
    "TR07": "Khowai",
    "TR08": "Unakoti (Kailashahar)",

    # ── Mizoram (MZ) ──
    "MZ01": "Aizawl",
    "MZ02": "Lunglei",
    "MZ03": "Champhai",
    "MZ04": "Kolasib",
    "MZ05": "Mamit",
    "MZ06": "Serchhip",
    "MZ07": "Lawngtlai",
    "MZ08": "Saitual",
    "MZ09": "Hnahthial",
    "MZ10": "Khawzawl",
    "MZ11": "Siaha",

    # ── Arunachal Pradesh (AR) ──
    "AR01": "Itanagar (Papum Pare)",
    "AR02": "East Kameng (Seppa)",
    "AR03": "West Kameng (Bomdila)",
    "AR04": "Tawang",
    "AR05": "Kurung Kumey (Koloriang)",
    "AR06": "Upper Subansiri (Daporijo)",
    "AR07": "Lower Subansiri (Ziro)",
    "AR08": "East Siang (Pasighat)",
    "AR09": "West Siang (Aalo)",
    "AR10": "Dibang Valley (Anini)",
    "AR11": "Lohit (Tezu)",
    "AR12": "Anjaw (Hawai)",
    "AR13": "Changlang",
    "AR14": "Tirap (Khonsa)",
    "AR15": "Lower Dibang Valley (Roing)",
    "AR16": "Upper Siang (Yingkiong)",
    "AR17": "Siang (Boleng)",
    "AR18": "Namsai",
    "AR19": "Pakke-Kessang",
    "AR20": "Lepa Rada",
    "AR21": "Shi Yomi",
    "AR22": "Kamle",

    # ── Sikkim (SK) ──
    "SK01": "East Sikkim (Gangtok)",
    "SK02": "West Sikkim (Gyalshing)",
    "SK03": "North Sikkim (Mangan)",
    "SK04": "South Sikkim (Namchi)",
    "SK05": "Pakyong",
    "SK06": "Soreng",

    # ── Puducherry (PY) ──
    "PY01": "Puducherry",
    "PY02": "Karaikal",
    "PY03": "Mahe",
    "PY04": "Yanam",

    # ── Andaman & Nicobar (AN) ──
    "AN01": "Port Blair (South Andaman)",
    "AN02": "Nicobar (Car Nicobar)",
    "AN03": "North & Middle Andaman (Mayabunder)",

    # ── Lakshadweep (LD) ──
    "LD01": "Kavaratti",
    "LD02": "Agatti",

    # ── Dadra & Nagar Haveli and Daman & Diu (DD / DN) ──
    "DD01": "Daman",
    "DD02": "Diu",
    "DD03": "Dadra & Nagar Haveli (Silvassa)",
    "DN01": "Silvassa",
    "DN02": "Daman",
    "DN03": "Diu",
}


def get_rto_info(state_code: str, district_num: str) -> str:
    """Look up the RTO office name from state code + zero-padded district number."""
    key = f"{state_code}{district_num.zfill(2)}"
    return RTO_CODES.get(key, f"RTO District {district_num}")

# ------------------------------------------------------------------
# OCR Character Correction Helpers
# ------------------------------------------------------------------

def correct_char_in_letter_zone(c: str) -> str:
    """Replace digit-lookalike characters with letters when in a letter zone."""
    MAP = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z', '4': 'A'}
    return MAP.get(c, c)

def correct_char_in_digit_zone(c: str) -> str:
    """Replace letter-lookalike characters with digits when in a digit zone."""
    MAP = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2', 'D': '0', 'Q': '0', 'A': '4', 'T': '1'}
    return MAP.get(c, c)

def apply_plate_corrections(plate: str) -> str:
    """
    Apply zone-aware OCR character corrections for Indian plates.
    Format: [2-letter state][2-digit district][1-3-letter series][4-digit number]
    Also handles BH-series format.
    """
    if len(plate) < 6:
        return plate

    # Zone 1: positions 0-1 → state code (letters)
    zone1 = ''.join(correct_char_in_letter_zone(c) for c in plate[:2])

    # Check for BH series after letter correction
    if zone1 == 'BH' and len(plate) >= 8:
        # BH + YYYY + 2digits + 1-2letters
        zone2 = ''.join(correct_char_in_digit_zone(c) for c in plate[2:6])  # year
        zone3 = ''.join(correct_char_in_digit_zone(c) for c in plate[6:8])  # 2 digits
        zone4 = ''.join(correct_char_in_letter_zone(c) for c in plate[8:])   # letters
        return zone1 + zone2 + zone3 + zone4

    # Try to find the trailing 1-4 digits from the end to be more robust
    valid_splits = []
    for i in range(max(0, len(plate)-4), len(plate)):
        suffix = ''.join(correct_char_in_digit_zone(c) for c in plate[i:])
        if re.match(r'^\d{1,4}$', suffix):
            prefix = plate[:i]
            if len(prefix) >= 4:
                z1 = ''.join(correct_char_in_letter_zone(c) for c in prefix[:2])
                z2 = ''.join(correct_char_in_digit_zone(c) for c in prefix[2:4])
                z3 = ''.join(correct_char_in_letter_zone(c) for c in prefix[4:])
                valid_splits.append((z1, z2, z3, suffix))
                
    if valid_splits:
        # Prefer the split with exactly 4 digits, or the longest suffix
        valid_splits.sort(key=lambda x: (len(x[3]) == 4, len(x[3])), reverse=True)
        z1, z2, z3, z4 = valid_splits[0]
        return z1 + z2 + z3 + z4

    # Fallback to basic positional splitting if trailing digits not found
    zone2 = ''.join(correct_char_in_digit_zone(c) for c in plate[2:4])
    rest = plate[4:]
    trailing_digits_match = re.search(r'\d+$', rest)
    if trailing_digits_match:
        digit_start = trailing_digits_match.start()
        zone3 = ''.join(correct_char_in_letter_zone(c) for c in rest[:digit_start])
        zone4 = ''.join(correct_char_in_digit_zone(c) for c in rest[digit_start:])
    else:
        if len(rest) >= 4:
            zone3 = ''.join(correct_char_in_letter_zone(c) for c in rest[:-4])
            zone4 = ''.join(correct_char_in_digit_zone(c) for c in rest[-4:])
        else:
            zone3 = ''.join(correct_char_in_letter_zone(c) for c in rest)
            zone4 = ""
    
    return zone1 + zone2 + zone3 + zone4

# ------------------------------------------------------------------
# Plate Parsing & Formatting
# ------------------------------------------------------------------

def format_plate_string(plate: str) -> str:
    """
    Format a cleaned plate string into standard Indian plate notation.
    e.g., MH12AB1234 → MH 12 AB 1234
    BH series: BH2021001AA → BH 2021 00 1AA
    Returns raw text if it doesn't match the expected format.
    """
    m = PLATE_REGEX.match(plate)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"

    bh = BH_PLATE_REGEX.match(plate)
    if bh:
        return f"{bh.group(1)} {bh.group(2)} {bh.group(3)} {bh.group(4)}"

    # Partial: at least add space after state code
    if len(plate) >= 2:
        return plate[:2] + ' ' + plate[2:]
    return plate

def clean_and_parse_plate(text: str):
    """
    Cleans raw OCR text:
      1. Strip to alphanumeric uppercase
      2. Remove leading IND country code
      3. Smart token merging for split reads
      4. Apply zone-aware OCR correction
      5. Validate against Indian plate regex (including BH series)
      6. Map state code → state name + RTO district office
    Returns (plate_number, state_code, state_name, is_valid_format, plate_series, rto_name).
    """
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

    # Strip leading IND country indicator
    if cleaned.startswith('IND'):
        cleaned = cleaned[3:]

    if not cleaned:
        return '', '', 'Unknown Registration', False, 'standard', 'Unknown RTO'

    # Apply corrections before validation
    corrected = apply_plate_corrections(cleaned)

    # Check BH series first
    bh_match = BH_PLATE_REGEX.match(corrected)
    if bh_match:
        state_code = 'BH'
        state = 'Bharat Series (Pan-India)'
        district_num = bh_match.group(3)  # 2-digit serial
        rto_name = f'Bharat Series — {bh_match.group(2)} Batch {district_num}'
        return corrected, state_code, state, True, 'bh', rto_name

    # Validate against standard Indian format
    m = PLATE_REGEX.match(corrected)
    is_valid = m is not None

    # If correction didn't help, also try the raw cleaned version
    if not is_valid:
        m_raw = PLATE_REGEX.match(cleaned)
        if m_raw:
            corrected = cleaned
            is_valid = True
            m = m_raw

    state_code = corrected[:2] if len(corrected) >= 2 else ''
    state = STATE_CODES.get(state_code, 'Unknown Registration')

    # Look up RTO district office
    if is_valid and m:
        district_num = m.group(2)  # e.g. '02' from KA02JK6254
        rto_name = get_rto_info(state_code, district_num)
    elif len(corrected) >= 4:
        district_num = corrected[2:4]
        rto_name = get_rto_info(state_code, district_num)
    else:
        rto_name = 'Unknown RTO'

    return corrected, state_code, state, is_valid, 'standard', rto_name

# ------------------------------------------------------------------
# OCR with Enhanced Preprocessing
# ------------------------------------------------------------------

def preprocess_clahe_sharp(crop: np.ndarray) -> np.ndarray:
    """
    Preprocessing path A — CLAHE + sharpen + bilateral filter.
    Best for: real-world plates with good contrast.
    """
    h, w = crop.shape[:2]

    min_height = 120
    if h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, (int(w * scale), min_height), interpolation=cv2.INTER_LANCZOS4)
    elif h < 200:
        crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Bilateral filter: PRESERVES text edges while smoothing noise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Unsharp mask to sharpen characters
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    return gray


def preprocess_binary(crop: np.ndarray) -> np.ndarray:
    """
    Preprocessing path B — Otsu's binarization.
    Best for: clean plates where characters are clearly defined.
    """
    h, w = crop.shape[:2]

    min_height = 120
    if h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, (int(w * scale), min_height), interpolation=cv2.INTER_LANCZOS4)
    elif h < 200:
        crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio < 0.4:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.dilate(binary, kernel, iterations=1)

    return binary


def rotate_crop(crop: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a crop by angle degrees around its center, keeping full image."""
    h, w = crop.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    return cv2.warpAffine(crop, M, (new_w, new_h), borderValue=(255, 255, 255))


def _extract_tokens(ocr_results: list) -> list[tuple[str, float]]:
    """
    From EasyOCR results, extract (token, confidence) pairs
    sorted top-to-bottom, left-to-right.
    Filters: skip 'IND', skip below threshold.
    """
    if not ocr_results:
        return []

    ocr_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    tokens = []
    for bbox, text, conf in ocr_results:
        token = text.strip().upper()
        if not token or token == 'IND':
            continue
        if conf < OCR_TOKEN_CONF_THRESHOLD:
            continue
        tokens.append((token, conf))
    return tokens


def _merge_tokens_smart(tokens: list[tuple[str, float]]) -> str:
    """
    Smart token merger: concatenates tokens and checks if merged string
    validates as an Indian plate before returning.
    Falls back to simple concatenation if no valid merge found.
    """
    if not tokens:
        return ''

    parts = [tok for tok, _ in tokens]
    # Try joining all
    joined = ''.join(parts)

    # If joined looks like a valid plate after correction, use it
    corrected = apply_plate_corrections(re.sub(r'[^A-Z0-9]', '', joined.upper()))
    if PLATE_REGEX.match(corrected) or BH_PLATE_REGEX.match(corrected):
        return corrected

    # Otherwise return raw joined (caller will clean further)
    return joined


def read_plate_ocr(crop: np.ndarray) -> str:
    """
    Multi-pass OCR strategy with angle rotation:
      Pass A: CLAHE + sharpen (0°)
      Pass B: Otsu binary (0°)
      Pass C: CLAHE + sharpen (+7° tilt)
      Pass D: CLAHE + sharpen (-7° tilt)
    Picks the result with highest total confidence × token count.
    Uses an allowlist restricted to plate characters to eliminate symbol misreads.
    """
    if crop is None or crop.size == 0:
        return ''

    ocr_kwargs = dict(
        detail=1,
        paragraph=False,
        allowlist=OCR_ALLOWLIST,
        text_threshold=0.45,
        low_text=0.25,
        decoder='greedy',
    )

    img_a = preprocess_clahe_sharp(crop)
    img_b = preprocess_binary(crop)
    img_c = preprocess_clahe_sharp(rotate_crop(crop, 7))
    img_d = preprocess_clahe_sharp(rotate_crop(crop, -7))

    results_a = reader.readtext(img_a, **ocr_kwargs)
    results_b = reader.readtext(img_b, **ocr_kwargs)
    results_c = reader.readtext(img_c, **ocr_kwargs)
    results_d = reader.readtext(img_d, **ocr_kwargs)

    tokens_a = _extract_tokens(results_a)
    tokens_b = _extract_tokens(results_b)
    tokens_c = _extract_tokens(results_c)
    tokens_d = _extract_tokens(results_d)

    # Score each pass: sum of (confidence × char_count) for all accepted tokens
    def score(tokens):
        return sum(conf * len(tok) for tok, conf in tokens)

    scores = {
        'a': (score(tokens_a), tokens_a),
        'b': (score(tokens_b), tokens_b),
        'c': (score(tokens_c), tokens_c),
        'd': (score(tokens_d), tokens_d),
    }

    best_score, best_tokens = max(scores.values(), key=lambda x: x[0])

    if not best_tokens:
        # Fallback: try again with lower thresholds on the CLAHE image
        fallback = reader.readtext(
            img_a,
            detail=1,
            paragraph=False,
            allowlist=OCR_ALLOWLIST,
            text_threshold=0.35,
            low_text=0.2,
            decoder='greedy',
        )
        fallback_tokens = []
        for bbox, text, conf in (fallback or []):
            token = text.strip().upper()
            if token and token != 'IND' and conf >= 0.2:
                fallback_tokens.append((token, conf))
        best_tokens = fallback_tokens

    # Use smart token merger to produce the best combined string
    raw_text = _merge_tokens_smart(best_tokens) if best_tokens else ''
    return raw_text

# ------------------------------------------------------------------
# Board Color Classification
# ------------------------------------------------------------------

def classify_plate_board(crop: np.ndarray) -> str:
    """
    Classifies the plate as:
      - 'White Board (Private)'
      - 'Yellow Board (Commercial)'
      - 'Green Board (Electric Vehicle)'
    using HSV color space analysis.
    """
    if crop is None or crop.size == 0:
        return "White Board (Private)"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Yellow HSV range
    lower_yellow = np.array([10, 50, 80])
    upper_yellow = np.array([34, 255, 255])

    # White HSV range
    lower_white = np.array([0, 0, 110])
    upper_white = np.array([180, 50, 255])

    # Green HSV range (EV plates)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([85, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    yellow_count = cv2.countNonZero(yellow_mask)
    white_count = cv2.countNonZero(white_mask)
    green_count = cv2.countNonZero(green_mask)

    total_pixels = crop.shape[0] * crop.shape[1]
    if total_pixels == 0:
        return "White Board (Private)"

    yellow_ratio = yellow_count / total_pixels
    green_ratio = green_count / total_pixels

    # Green EV check (takes priority over yellow, both can overlap)
    if green_ratio > 0.15 and green_count > (white_count * 0.6):
        return "Green Board (Electric Vehicle)"
    elif yellow_ratio > 0.12 and yellow_count > (white_count * 0.8):
        return "Yellow Board (Commercial)"
    else:
        return "White Board (Private)"

def check_is_hsrp(crop: np.ndarray) -> bool:
    """
    Checks if a plate is an HSRP plate by looking for the blue 'IND' 
    patch on the far-left side of the image.
    """
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    # Look only at the left 20% of the image
    left_slice = crop[:, :int(w * 0.2)]

    if left_slice.size == 0:
        return False

    hsv = cv2.cvtColor(left_slice, cv2.COLOR_BGR2HSV)

    # Blue HSV range for the IND patch
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_count = cv2.countNonZero(blue_mask)
    total_pixels = left_slice.shape[0] * left_slice.shape[1]

    if total_pixels == 0:
        return False

    blue_ratio = blue_count / total_pixels

    # If more than 1.5% of the left slice is true blue, it's highly likely HSRP
    return blue_ratio > 0.015

# ------------------------------------------------------------------
# Core Detection Pipeline
# ------------------------------------------------------------------

def process_image(img: np.ndarray):
    """
    Runs the full pipeline:
      1. YOLO detection with confidence threshold filtering
      2. Plate crop preprocessing
      3. Multi-pass EasyOCR with smart token merging
      4. Zone-aware OCR correction + Indian/BH plate validation
      5. Board color classification (white/yellow/green)
      6. Annotated image rendering
    Returns (annotated_img, detections).
    """
    h, w, _ = img.shape
    results = model(img, verbose=False)
    detections = []
    annotated_img = img.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])

            if conf < YOLO_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Add 5% padding to improve OCR on edge characters
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            crop = img[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            board_type = classify_plate_board(crop)
            is_hsrp = check_is_hsrp(crop)

            _, crop_buffer = cv2.imencode('.jpg', crop)
            crop_base64 = f"data:image/jpeg;base64,{base64.b64encode(crop_buffer).decode('utf-8')}"
            raw_text = read_plate_ocr(crop)

            plate_number = ''
            state_code = ''
            state_name = 'Unknown Registration'
            is_valid_format = False
            plate_series = 'standard'
            rto_name = 'Unknown RTO'

            if raw_text:
                plate_number, state_code, state_name, is_valid_format, plate_series, rto_name = clean_and_parse_plate(raw_text)

            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "raw_text": raw_text,
                "plate_number": plate_number,
                "plate_formatted": format_plate_string(plate_number) if plate_number else '',
                "state_code": state_code,
                "state_name": state_name,
                "board_type": board_type,
                "is_hsrp": is_hsrp,
                "is_valid_format": is_valid_format,
                "plate_series": plate_series,
                "rto_name": rto_name,
                "crop_image": crop_base64
            })

            # --- Draw bounding box and label ---
            if "Green" in board_type:
                color = (0, 200, 100)   # Green for EV
            elif "Yellow" in board_type:
                color = (0, 215, 255)   # Amber-yellow for commercial
            else:
                color = (0, 255, 0)     # Green for private

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)

            if "Green" in board_type:
                board_short = "EV"
            elif "Yellow" in board_type:
                board_short = "Commercial"
            else:
                board_short = "Private"

            display_plate = format_plate_string(plate_number) if plate_number else 'Plate'
            hsrp_tag = " [HSRP]" if is_hsrp else ""
            label = f"{display_plate}{hsrp_tag} ({board_short})"

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_img,
                (x1, y1 - text_size[1] - 15),
                (x1 + max(text_size[0] + 10, 180), y1),
                color,
                -1
            )
            cv2.putText(
                annotated_img,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

    return annotated_img, detections


# ------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------

class FrameData(BaseModel):
    image: str  # Base64 encoded JPEG data URL or raw base64 string


class ExportData(BaseModel):
    detections: list  # List of detection dicts to export as CSV


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/")
def get_index():
    return FileResponse("static/index.html")


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        annotated_img, detections = process_image(img)

        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "detections": detections,
            "image": f"data:image/jpeg;base64,{img_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-frame")
async def detect_frame(data: FrameData):
    try:
        img_data = data.image
        if "," in img_data:
            img_data = img_data.split(",")[1]

        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image frame data")

        annotated_img, detections = process_image(img)

        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "detections": detections,
            "image": f"data:image/jpeg;base64,{img_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export-csv")
async def export_csv(data: ExportData):
    """
    Accept a list of detection records and return a downloadable CSV file.
    Columns: plate_number, plate_formatted, state_code, state_name, board_type,
             confidence, is_valid_format, plate_series, raw_text, timestamp
    """
    try:
        output = io.StringIO()
        fieldnames = [
            'plate_formatted', 'plate_number', 'state_code', 'state_name',
            'board_type', 'plate_series', 'confidence', 'is_valid_format',
            'raw_text', 'timestamp'
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for det in data.detections:
            row = {
                'plate_formatted': det.get('plate_formatted', ''),
                'plate_number': det.get('plate_number', ''),
                'state_code': det.get('state_code', ''),
                'state_name': det.get('state_name', ''),
                'board_type': det.get('board_type', ''),
                'plate_series': det.get('plate_series', 'standard'),
                'confidence': f"{float(det.get('confidence', 0)) * 100:.1f}%",
                'is_valid_format': 'Yes' if det.get('is_valid_format') else 'No',
                'raw_text': det.get('raw_text', ''),
                'timestamp': det.get('timestamp', ''),
            }
            writer.writerow(row)

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=swiftplate_history.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
