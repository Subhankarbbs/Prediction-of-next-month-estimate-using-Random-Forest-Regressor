import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set visual aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_datasets(base_path="."):
    files_map = {
        'Enrolment': 'api_data_aadhar_enrolment_*.csv',
        'Demographic': 'api_data_aadhar_demographic_*.csv',
        'Biometric': 'api_data_aadhar_biometric_*.csv'
    }
    
    datasets = {}
    print("Loading datasets...")
    for category, pattern in files_map.items():
        full_pattern = os.path.join(base_path, pattern)
        files = glob.glob(full_pattern)
        
        df_list = []
        for file in files:
            try:
                temp = pd.read_csv(file)
                df_list.append(temp)
            except Exception as e:
                print(f"Skipped {file}: {e}")
        
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
                # Date Features
                df['Month'] = df['date'].dt.month_name()
                df['DayOfWeek'] = df['date'].dt.day_name()
                df['IsWeekend'] = df['date'].dt.dayofweek.isin([5, 6])
            datasets[category] = df
            print(f"  -> Loaded {category}: {len(df)} records")
        else:
            print(f"  -> No files found for {category}")
            datasets[category] = pd.DataFrame() 

    return datasets['Enrolment'], datasets['Demographic'], datasets['Biometric']

# ==========================================
# FINAL COMPREHENSIVE CLEANING FUNCTION
# ==========================================
def clean_data(df):
    if df.empty: return df
    
    # This Regex removes Chinese chars, emojis, and weird symbols.
    # It keeps only standard ASCII (English text, numbers, basic symbols)
    df['district'] = df['district'].astype(str).replace(r'[^\x00-\x7F]+', '', regex=True)
    # 1. Basic Cleanup: Remove '*' and trim whitespace
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).str.replace('*', '', regex=False).str.strip()
        # Fix specific merged name
        df.loc[df['district'] == 'ManendragarhChirmiriBharatpur', 'district'] = 'Manendragarh-Chirmiri-Bharatpur'

    # 2. Standardize State Names (Fixing Typos)
    state_map = {
        'WEST BENGAL': 'West Bengal', 'WESTBENGAL': 'West Bengal', 
        'West bengal': 'West Bengal', 'Westbengal': 'West Bengal', 
        'West Bengli': 'West Bengal', 'west Bengal': 'West Bengal',
        'West  Bengal': 'West Bengal', 'West Bangal': 'West Bengal',
        'odisha': 'Odisha', 'ODISHA': 'Odisha', 'Orissa': 'Odisha',
        'andhra pradesh': 'Andhra Pradesh',
        'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
        'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'Dadra and Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'Daman and Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'Pondicherry': 'Puducherry', 'Uttaranchal': 'Uttarakhand',
        'Tamilnadu': 'Tamil Nadu', 'Chhatisgarh': 'Chhattisgarh',
        'Telanana': 'Telangana', # Fix typo
        '100000': 'Unknown',
        'Jaipur': 'Rajasthan', 'Nagpur': 'Maharashtra', 'Darbhanga': 'Bihar',
        'Madanapalle': 'Andhra Pradesh', 'BALANAGAR': 'Telangana',
        'Puttenahalli': 'Karnataka', 'Raja Annamalai Puram': 'Tamil Nadu'
    }
    
    if 'state' in df.columns:
        df['state'] = df['state'].replace(state_map)
    # Fix: Move Kamrup from Meghalaya to Assam
    df.loc[(df['state'] == 'Meghalaya') & (df['district'] == 'Kamrup'), 'state'] = 'Assam'
    
    # 3. RE-ASSIGNING DISTRICTS TO CORRECT STATES
    # Moves Telangana districts out of Andhra Pradesh, etc.
    ts_districts = [
        'Adilabad', 'Hyderabad', 'K.V RANGAEEDDY', 'Karimnagar', 'Khammam', 
        'Mahabubnagar', 'Medak', 'Nalgonda', 'Nizamabad', 'Rangareddi', 
        'Warangal', 'Warangal(urban)', 'Warangal(rural)', 'Jagitial', 
        'Jangaon', 'Jayashankar Bhupalpally', 'Jogulamba Gadwal', 
        'Kamareddy', 'Komaram Bheem Asifabad', 'Mahabubabad', 'Mancherial', 
        'Medchal–Malkajgiri', 'Mulugu', 'Nagarkurnool', 'Narayanpet', 
        'Nirmal', 'Peddapalli', 'Rajanna Sircilla', 'Sangareddy', 
        'Siddipet', 'Suryapet', 'Vikarabad', 'Wanaparthy', 'Yadadri'
    ]
    df.loc[(df['state'] == 'Andhra Pradesh') & (df['district'].isin(ts_districts)), 'state'] = 'Telangana'
    df.loc[(df['state'] == 'Chandigarh') & (df['district'].isin(['Mohali', 'Rupnagar'])), 'state'] = 'Punjab'
    df.loc[(df['state'] == 'Puducherry') & (df['district'].isin(['Cuddalore', 'Viluppuram'])), 'state'] = 'Tamil Nadu'
    df.loc[(df['state'] == 'Andhra Pradesh') & (df['district'] == 'Karim Nagar'), 'state'] = 'Telangana'
    df.loc[(df['state'] == 'Andhra Pradesh') & (df['district'] == 'Ranga Reddy'), 'state'] = 'Telangana'
    df.loc[(df['state'] == 'Andhra Pradesh') & (df['district'] == 'Mahabub Nagar'), 'state'] = 'Telangana'
    df.loc[(df['state'] == 'Jammu and Kashmir') & (df['district'] == 'Leh'), 'state'] = 'Ladakh'
    df.loc[(df['state'] == 'Jammu and Kashmir') & (df['district'] == 'Kargil'), 'state'] = 'Ladakh'
    df.loc[(df['state'] == 'Jammu and Kashmir') & (df['district'] == 'Leh (ladakh)'), 'state'] = 'Ladakh'

    # 4. RENAME DISTRICTS (Standardizing Names)
    district_map = {
        # --- WEST BENGAL MERGES ---
        'Bardhaman': 'Purba Bardhaman', 'Burdwan': 'Purba Bardhaman',
        'CoochBehar': 'Cooch Behar', 'Coochbehar': 'Cooch Behar',
        'South Dinajpur': 'Dakshin Dinajpur', 'North Dinajpur': 'Uttar Dinajpur',
        'Domjur': 'Howrah', 'East Midnapore': 'Purba Medinipur',
        'West Medinipur': 'Paschim Medinipur', 'Medinipur': 'Paschim Medinipur',
        'Medinipur West': 'Paschim Medinipur', 'Naihati Anandbazar': 'North 24 Parganas',
        'Naihati Anandabazar': 'North 24 Parganas', 'Nortg 24 praganas': 'North 24 Parganas',
        'South 24 praganas': 'South 24 Parganas',
        
        # --- UP / UTTARAKHAND ---
        'Garhwal': 'Pauri Garhwal', 'Allahabad': 'Prayagraj', 'Faizabad': 'Ayodhya',
        'Sant Ravidas Nagar': 'Bhadohi', 'Sant Ravidas Nagar Bhadohi': 'Bhadohi',
        'Baghpat': 'Baghpat', 'Barabanki': 'Barabanki', 'Kushinagar': 'Kushinagar',
        'Maharajganj': 'Maharajganj', 'Raebareli': 'Raebareli', 
        'Shravasti': 'Shravasti', 'Siddharthnagar': 'Siddharthnagar',
        
        # --- TELANGANA ---
        'IDPL COLONY': 'Unknown', 'Jagitial': 'Jagtial', 'Jangaon': 'Jangaon',
        'K.v. Rangareddy': 'Ranga Reddy', 'K.V RANGAEEDDY': 'Ranga Reddy',
        'Rangareddy': 'Ranga Reddy', 'Komaram Bheem Asifabad': 'Komaram Bheem',
        'Medchal–Malkajgiri': 'Medchal Malkajgiri', 'Warangal(urban)': 'Hanumakonda',
        'Warangal(rural)': 'Warangal', 'Yadadri': 'Yadadri Bhuvanagiri',
        
        # --- OTHERS ---
        'Andamans':'South Andaman','Nicobars': 'Nicobar', 'Ananthapur':'Ananthapuramu', 
        'Cuddapah': 'YSR Kadapa', 'Chittoor': 'Chittoor', 
        'Shiyomi': 'Shi Yomi', 'Pakke-Kessang': 'Pakke Kessang', 'Kra-Daadi': 'Kra Daadi',
        'Kamrup Metropolitan': 'Kamrup Metro', 'South Salmara-Mankachar': 'South Salmara Mankachar',
        'Chhatrapati Sambhajinagar': 'Chatrapati Sambhaji Nagar', # Context specific fix for Bihar
        'Kaimur': 'Kaimur (Bhabhua)', 'Munger': 'Munger', 'Samastipur': 'Samastipur',
        'Purnia': 'Purnia', 'Purba Champaran': 'East Champaran', 
        'Near university': 'Unknown',
        'Dantewada': 'Dantewada', 'Janjgir-Champa': 'Janjgir-Champa',
        'Gaurela-Pendra-Marwahi': 'Gaurella-Pendra-Marwahi', 
        'Mohla-Manpur-Ambagarh Chowki': 'Mohla-Manpur-Ambagarh Chowki',
        'Ahmadabad': 'Ahmedabad', 'Banas Kantha': 'Banaskantha',
        'Panch Mahals': 'Panchmahal', 'Sabar Kantha': 'Sabarkantha',
        'Surendra Nagar': 'Surendranagar', 'Yamuna Nagar': 'Yamunanagar',
        'Lahaul and Spiti': 'Lahaul and Spiti', 'Lahul & Spiti': 'Lahaul and Spiti',
        'Bandipore': 'Bandipora', 'Bandipur': 'Bandipora', 
        'Poonch': 'Poonch', 'Punch': 'Poonch', 'Rajauri': 'Rajouri', 
        'Shopian': 'Shopian', 'Shupiyan': 'Shopian', 'Udhampur': 'Udhampur',
        'East Singhbum': 'East Singhbhum', 'Hazaribag': 'Hazaribagh',
        'Koderma': 'Kodarma', 'Pakaur': 'Pakur', 'Palamau': 'Palamu',
        'Sahebganj': 'Sahibganj', 'Seraikela-kharsawan': 'Seraikela Kharsawan',
        'Pashchimi Singhbhum': 'West Singhbhum',
        '5th cross': 'Bengaluru Urban', 'Bangalore': 'Bengaluru Urban',
        'Bengaluru': 'Bengaluru Urban', 'Bijapur': 'Vijayapura',
        'Chickmagalur': 'Chikkamagaluru',
        'Chikmagalur': 'Chikkamagaluru', 'Davanagere': 'Davangere',
        'Hasan': 'Hassan', 'Mysore': 'Mysuru', 'Ramanagar': 'Ramanagara',
        'Shimoga': 'Shivamogga', 'Tumkur': 'Tumakuru', 'Yadgir': 'Yadgir',
        'ANGUL': 'Angul', 'Boudh': 'Boudh', 'Jajpur': 'Jajpur',
        'Jagatsinghpur': 'Jagatsinghpur', 'Khordha': 'Khordha', 
        'Nabarangpur': 'Nabarangpur', 'Subarnapur': 'Subarnapur',
        'Muktsar': 'Sri Muktsar Sahib', 'Nawanshahr': 'Shaheed Bhagat Singh Nagar',
        'S.A.S Nagar': 'SAS Nagar (Mohali)', 'Ferozepur': 'Ferozepur',
        'Chittorgarh': 'Chittorgarh', 'Dholpur': 'Dholpur', 'Jalore': 'Jalore',
        'Jhunjhunu': 'Jhunjhunu', 'Near meera hospital': 'Unknown',
        'East Sikkim': 'Gangtok', 'North Sikkim': 'Mangan (North)',
        'South Sikkim': 'Namchi (South)', 'West Sikkim': 'Gyalshing (West)',
        'Near Dhyana Ashram': 'Unknown', 'Thiruvallur': 'Tiruvallur',
        'Thiruvarur': 'Tiruvarur', 'Tuticorin': 'Thoothukkudi',
        'Tirupattur': 'Tirupattur',
        'Aurangabad': 'Chhatrapati Sambhajinagar', 'Osmanabad': 'Dharashiv',
        'Ahmednagar': 'Ahilyanagar', 'Gurgaon': 'Gurugram',
        'Budaun': 'Badaun', 'Bulandshahar': 'Bulandshahr',
        'Mumbai( Sub Urban )': 'Mumbai Suburban',
        'Chhatrapati Sambhaji Nagar': 'Chhatrapati Sambhajinagar',
        'Mahbubnagar':'Mahabub Nagar','Visakhapatanam':'Visakhapatnam','Y. S. R':'YSR Kadapa',
        'chittoor':'Chittoor','rangareddi':'Ranga Reddy','K.V.Rangareddy':'Ranga Reddy',
        
        #ANDHRA PRADESH
        'Kadiri Road':'Sri Sathya Sai','Anantapur':'Ananthapuramu','Nellore':'Sri Potti Sriramulu Nellore',
        'Spsr Nellore':'Sri Potti Sriramulu Nellore',

        #ASSAM
        'Karimganj':'Sribhumi','North Cachar Hills':'Dima Hasao','Sibsagar':'Sivasagar',
        'Tamulpur District':'Tamulpur',

        #BIHAR
        'Aurangabad(BH)':'Aurangabad','Aurangabad(bh)':'Aurangabad','Bhabua':'Kaimur (Bhabua)',
        'Monghyr':'Munger','Near University Thana':'Darbhanga','Purnea':'Purnia',
        'Samstipur':'Samastipur','Sheikpura':'Sheikhpura','Pashchim Champaran':'West Champaran',
        'East Champaran':'Purbi Champaran','West Champaran':'Paschim Champaran',

        #CHHATTISGARH
        'Dakshin Bastar Dantewada':'Dantewada','Gaurela-pendra-marwahi':'Gaurella Pendra Marwahi',
        'Janjgir - Champa':'Janjgir-champa','Janjgir Champa':'Janjgir-champa',
        'Manendragarh鈥揅hirmiri鈥揃haratpur':'Manendragarh-Chirmiri-Bharatpur',
        'Mohalla-Manpur-Ambagarh Chowki':'Mohla-Manpur-Ambagarh Chouki','Vijayapura':'Bijapur',

        #Dadra and Nagar Haveli and Daman and Diu
        'Dadra & Nagar Haveli':'Dadra and Nagar Haveli','Dadra And Nagar Haveli':'Dadra and Nagar Haveli',
        
        #DELHI
        'Najafgarh':'South West Delhi','North East':'North East Delhi',

        #GOA
        'Bardez':'North Goa','Bicholim':'North Goa','Tiswadi':'North Goa',

        #GUJARAT
        'Panchmahals':'Panchmahal','Dohad':'Dahod','The Dangs':'Dangs','Dang':'Dangs',

        #HARYANA
        'Akhera':'Rewari','Mewat':'Nuh',

        #HIMACHAL PRADESH
        'Lahul and Spiti':'Lahaul and Spiti',

        #JAMMU AND KASHMIR
        '?':'Jammu','punch':'Poonch','udhampur':'Udhampur','Badgam':'Budgam',

        #JHARKHAND
        'Seraikela Kharsawan':'Seraikela-Kharsawan','Purbi Singhbhum':'West Singhbhum',
        'Seraikela Kharsawan':'Seraikela-Kharsawan',

        #KARNATAKA
        'Bengaluru Rural':'Bangalore Rural','Bijapur(KAR)':'Vijayapura','Chamarajanagara':'Chamrajanagar',
        'Chamrajnagar':'Chamrajanagar','yadgir':'Yadgir','Bellary':'Ballari','Belgaum':'Belagavi',
        'Gulbarga':'Kalaburagi','Ramanagara':'Bengaluru South',

        #KERALA
        'Kasargod':'Kasaragod',

        #LADAKH
        'Leh (ladakh)':'Leh',

        #MADHYA PRADESH
        'Ashoknagar':'Ashok Nagar','Narsimhapur':'Narsinghpur','Hoshangabad':'Narmadapuram',
        'East Nimar':'Khandwa','West Nimar':'Khargone',

        #MAHARASHTRA
        'Ahmadnagar':'Ahilyanagar','Ahmed Nagar':'Ahilyanagar','Aurangabad':'Chatrapati Sambhaji Nagar',
        'Bid':'Beed','Buldana':'Buldhana','Dist : Thane':'Thane','Gondiya':'Gondia',
        'Near Uday nagar NIT garden':'Nagpur','Raigarh':'Raigad','Raigarh(MH)':'Raigad',

        #MEGHALAYA
        'Jaintia Hills':'West Jaintia Hills',

        #MIZORAM
        'Mammit':'Mamit',

        #ODISHA
        'ANUGUL':'Angul','Anugal':'Angul','Anugul':'Angul','BALANGIR':'Balangir','Baleshwar':'Baleswar',
        'Balianta':'Khordha','Baudh':'Boudh','Bhadrak(R)':'Bhadrak','JAJPUR':'Jajpur','Jajapur':'Jajpur',
        'Jagatsinghpur':'Jagatsinghapur','Khorda':'Khordha','NAYAGARH':'Nayagarh','NUAPADA':'Nuapada',
        'Nabarangapur':'Nabarangpur','Sonapur':'Subarnapur','Sundergarh':'Sundargarh','jajpur':'Jajpur',

        #PUDUCHERRY
        'Pondicherry':'Puducherry','Yanam':'Puducherry',

        #PUNJAB
        'Ferozepur':'Firozpur','Mohali':'SAS Nagar (Mohali)','S.A.S Nagar(Mohali)':'SAS Nagar (Mohali)',
        'SAS Nagar':'SAS Nagar (Mohali)',
        
        #RAJASTHAN
        'Chittaurgarh':'Chittorgarh','Dhaulpur':'Dholpur','Jalor':'Jalore','Jhunjhunun':'Jhunjhunu',

        #SIKKIM
        'South':'Namchi (South)','West':'Gyalshing (West)','North':'Mangan (North)',
        'West':'Gyalshing (West)','Mangan':'Mangan (North)','Namchi':'Namchi (South)',

        #TAMIL NADU
        'Kanchipuram':'Kancheepuram','Kanniyakumari':'Kanyakumari','Tirupattur':'Tirupathur',
        'Villupuram':'Viluppuram',

        #TELANGANA
        'Jangoan':'Jangaon','Medchal Malkajgiri':'Medchal-Malkajgiri','Medchal-malkajgiri':'Medchal-Malkajgiri',
        'Medchal芒聢聮malkajgiri':'Medchal-Malkajgiri','Medchal鈭抦alkajgiri':'Medchal-Malkajgiri',
        'Rangareddi':'Ranga Reddy','Warangal (urban)':'Hanumakonda','Warangal Urban':'Hanumakonda',
        'Warangal Rural':'Warangal','Yadadri.':'Yadadri Bhuvanagiri','Medchalmalkajgiri':'Medchal-Malkajgiri',
        'Mahabub Nagar':'Mahabubnagar','Karim Nagar':'Karimnagar',

        #UTTAR PRADESH
        'Baghpat':'Bagpat','Bara Banki':'Barabanki','Jyotiba Phule Nagar':'Amroha','Kheri':'Lakhimpur Kheri',
        'Kushi Nagar':'Kushinagar','Mahrajganj':'Maharajganj','Raebareli':'Rae Bareli','Shrawasti':'Shravasti',
        'Siddharth Nagar':'Siddharthnagar',

        #UTTARAKHAND
        'Hardwar':'Haridwar',

        #WEST BENGAL
        '24 Paraganas North':'North 24 Parganas','24 Paraganas South':'South 24 Parganas',
        'Bally Jagachha':'Howrah','Darjiling':'Darjeeling','Dinajpur Dakshin':'Dakshin Dinajpur',
        'Dinajpur Uttar':'Uttar Dinajpur','East Midnapur':'Purba Medinipur','East midnapore':'Purba Medinipur',
        'HOOGHLY':'Hooghly','Hooghiy':'Hooghly','HOWRAH':'Howrah','Haora':'Howrah','Hawrah':'Howrah',
        'Hugli':'Hooghly','KOLKATA':'Kolkata','Koch Bihar':'Cooch Behar','MALDA':'Malda','Maldah':'Malda',
        'NADIA':'Nadia','North Twenty Four Parganas':'North 24 Parganas','Puruliya':'Purulia',
        'South  Twenty Four Parganas':'South 24 Parganas','South 24 Pargana':'South 24 Parganas',
        'South 24 pargana':'South 24 Parganas','South 24 parganas':'South 24 Parganas',
        'South Twenty Four Parganas':'South 24 Parganas','South DumDum(M)':'North 24 Parganas',
        'West Midnapore':'Paschim Medinipur','east midnapore':'Purba Medinipur',
        'hooghly':'Hooghly','nadia':'Nadia'

    }
    
    if 'district' in df.columns:
        df['district'] = df['district'].replace(district_map)
        df = df[df['district'] != 'Unknown']

    # =======================================================
    # NEW FIX: REMOVE '100000' AND 'UNKNOWN' STATES
    # =======================================================
    if 'state' in df.columns:
        # Remove rows where state is 'Unknown' (which we mapped from 100000)
        df = df[df['state'] != 'Unknown']
        # Remove any remaining '100000' if they missed the mapping
        df = df[df['state'] != '100000']
    
    if 'district' in df.columns:
        # Remove rows where district is specifically '100000'
        df = df[df['district'] != '100000']
    # =======================================================
    return df

# ======================================================
# SNIPPET: EXPORT MONTHLY DATA (New Function)
# ======================================================
def export_monthly_data(enrol, demo, bio):
    print("\n[Action] Generating Monthly Age-Group Time Series...")

    def aggregate_monthly(df, prefix, cols):
        if 'date' not in df.columns: return pd.DataFrame()
        temp = df.copy()
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(temp['date']):
             temp['date'] = pd.to_datetime(temp['date'], format='%d-%m-%Y', errors='coerce')
        
        temp['YearMonth'] = temp['date'].dt.to_period('M').astype(str)
        grouped = temp.groupby(['state', 'district', 'YearMonth'])[cols].sum().reset_index()
        new_cols = {c: f"{prefix}_{c}" for c in cols}
        grouped = grouped.rename(columns=new_cols)
        return grouped

    # 1. Process Enrolment
    print("   -> Processing Enrolment trends...")
    e_monthly = aggregate_monthly(enrol, 'Enrol', ['age_0_5', 'age_5_17', 'age_18_greater'])
    
    # 2. Process Demographic
    print("   -> Processing Demographic trends...")
    d_monthly = aggregate_monthly(demo, 'Demo', ['demo_age_5_17', 'demo_age_17_'])
    
    # 3. Process Biometric
    print("   -> Processing Biometric trends...")
    b_monthly = aggregate_monthly(bio, 'Bio', ['bio_age_5_17', 'bio_age_17_'])
    
    # 4. Merge
    master_ts = pd.merge(e_monthly, d_monthly, on=['state', 'district', 'YearMonth'], how='outer').fillna(0)
    master_ts = pd.merge(master_ts, b_monthly, on=['state', 'district', 'YearMonth'], how='outer').fillna(0)
    
    # 5. Save
    master_ts.to_csv('aadhaar_monthly_district_trends.csv', index=False)
    print(f"   -> Success! Saved monthly trends to 'aadhaar_monthly_district_trends.csv'")
    return master_ts

# ==========================================
# 3. METRIC CALCULATION (THE 20 RELATIONS)
# ==========================================
def calculate_metrics(enrol, demo, bio):
    print("\nCalculating Analytical Metrics...")
    
    # Aggregate
    e_grp = enrol.groupby(['state', 'district'])[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
    d_grp = demo.groupby(['state', 'district'])[['demo_age_5_17', 'demo_age_17_']].sum()
    b_grp = bio.groupby(['state', 'district'])[['bio_age_5_17', 'bio_age_17_']].sum()
    
    df = e_grp.join([d_grp, b_grp], how='outer').fillna(0)
    
    # Totals
    df['Enrol_Total'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    df['Update_Total'] = df['demo_age_5_17'] + df['demo_age_17_'] + df['bio_age_5_17'] + df['bio_age_17_']
    df['Grand_Total'] = df['Enrol_Total'] + df['Update_Total']
    
    # Filter Low Volume
    df = df[df['Grand_Total'] > 500].copy()

    # --- CATEGORY A: OPERATIONAL ---
    # 1. UER (Update to Enrolment Ratio)
    df['R1_UER'] = df['Update_Total'] / (df['Enrol_Total'] + 1)
    
    # 2. Bio-to-Demo Ratio
    df['R2_Bio_Demo_Ratio'] = (df['bio_age_5_17'] + df['bio_age_17_']) / (df['demo_age_5_17'] + df['demo_age_17_'] + 1)
    
    # --- CATEGORY B: DEMOGRAPHIC ---
    # 3. Catch-up Index (Missed Births)
    df['R3_Catch_Up_Index'] = df['age_5_17'] / (df['age_0_5'] + 1)
    
    # 4. Adult Entry Rate (Migration/Fraud)
    df['R4_Adult_Entry_Rate'] = df['age_18_greater'] / (df['Enrol_Total'] + 1)
    
    # 5. Child Enrolment Rate
    df['R5_Child_Share'] = df['age_0_5'] / (df['Enrol_Total'] + 1)

    # --- CATEGORY C: ANOMALY ---
    # 6. Ghost Village Proxy (High Enrolment, Zero Updates)
    df['R6_Ghost_Proxy'] = df['Enrol_Total'] / (df['Update_Total'] + 1)
    
    # 7. Adult Z-Score (Statistical Anomaly)
    mean_adult = df['R4_Adult_Entry_Rate'].mean()
    std_adult = df['R4_Adult_Entry_Rate'].std()
    df['R7_Adult_ZScore'] = (df['R4_Adult_Entry_Rate'] - mean_adult) / (std_adult + 1e-5)

    return df

# ==========================================
# 4. VISUALIZATION FUNCTIONS
# ==========================================

# A. Radar Chart (Weekend Gap)
def plot_radar_chart(enrol, demo, bio):
    # Combine data for daily volume
    master = pd.concat([
        enrol[['date', 'age_5_17']].rename(columns={'age_5_17':'vol'}),
        demo[['date', 'demo_age_5_17']].rename(columns={'demo_age_5_17':'vol'}),
        bio[['date', 'bio_age_5_17']].rename(columns={'bio_age_5_17':'vol'})
    ])
    master['Day'] = master['date'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_vol = master.groupby('Day')['vol'].sum().reindex(days_order).fillna(0)
    
    # Radar Logic
    values = daily_vol.values.flatten().tolist()
    values += values[:1] 
    angles = [n / float(len(days_order)) * 2 * pi for n in range(len(days_order))]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], days_order, color='grey', size=10)
    ax.plot(angles, values, linewidth=1, linestyle='solid', color='blue')
    ax.fill(angles, values, 'b', alpha=0.1)
    plt.title('Weekly Activity Radar: The "Sunday Service Gap"', size=15, y=1.1)
    plt.tight_layout()
    plt.savefig('vis_radar_weekly.png')
    plt.show()

# B. Stacked Bar (Digital vs Physical)
def plot_digital_physical(metrics_df):
    # Sort by Total Volume for Top 10 States
    top_states = metrics_df.groupby('state')[['Update_Total', 'demo_age_17_', 'bio_age_17_']].sum()
    top_states = top_states.sort_values('Update_Total', ascending=False).head(10)
    
    # Calc Ratios
    top_states['Demo_Share'] = top_states['demo_age_17_'] / top_states['Update_Total']
    top_states['Bio_Share'] = top_states['bio_age_17_'] / top_states['Update_Total']
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_states.index, top_states['Demo_Share'], label='Demographic (Online/Easy)', color='#4c72b0')
    plt.bar(top_states.index, top_states['Bio_Share'], bottom=top_states['Demo_Share'], label='Biometric (Physical/Hard)', color='#55a868')
    plt.title('Digital Maturity: Demographic vs Biometric Update Composition', fontsize=14)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('vis_stacked_split.png')
    plt.show()

# C. Seasonality Line Chart
def plot_seasonality(enrol):
    monthly = enrol.groupby('Month')['age_5_17'].sum().reindex([
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    plt.figure(figsize=(12, 5))
    monthly.plot(marker='o', linestyle='-', color='purple')
    plt.title('The "School Pulse": Monthly Seasonality of Child Enrolments', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vis_seasonality.png')
    plt.show()

# ==========================================
# 5. EXECUTION MAIN
# ==========================================
# 1. Load
enrol_df, demo_df, bio_df = load_datasets()

# 2. Clean
enrol_df = clean_data(enrol_df)
demo_df = clean_data(demo_df)
bio_df = clean_data(bio_df)

# ==========================================
# GENERATE MONTHLY TRENDS CSV
# ==========================================
export_monthly_data(enrol_df, demo_df, bio_df)
# ==========================================

# 3. Calculate Metrics
metrics_df = calculate_metrics(enrol_df, demo_df, bio_df)

# 4. Generate Visualizations
plot_radar_chart(enrol_df, demo_df, bio_df)
plot_digital_physical(metrics_df)
plot_seasonality(enrol_df)

# 5. Print Top Insights
print("\n" + "="*50)
print("AADHAAR 360 ANALYSIS REPORT")
print("="*50)
print("\n[INSIGHT 1] Top 10 'Maintenance Only' Districts (High UER):")
print(metrics_df.sort_values('R1_UER', ascending=False)['R1_UER'].head(10))

print("\n[INSIGHT 2] Top 10 'Missing Births' Districts (High Catch-up Index):")
print(metrics_df.sort_values('R3_Catch_Up_Index', ascending=False)['R3_Catch_Up_Index'].head(10))

print("\n[INSIGHT 3] Top 10 Anomalous Adult Enrolments (Potential Fraud):")
print(metrics_df.sort_values('R7_Adult_ZScore', ascending=False)[['R4_Adult_Entry_Rate', 'R7_Adult_ZScore']].head(10))
# ==========================================
# PHASE 2: REGIONAL & VOLATILITY EXPANSION
# ==========================================
def calculate_phase2_metrics(enrol_df):
    print("\nCalculating Phase 2 Metrics (Regional & Stability)...")
    
    # 1. Regional Mapping
    region_map = {
        'Jammu and Kashmir': 'North', 'Himachal Pradesh': 'North', 'Punjab': 'North', 
        'Uttarakhand': 'North', 'Haryana': 'North', 'Delhi': 'North', 'Uttar Pradesh': 'North',
        'Bihar': 'East', 'Jharkhand': 'East', 'West Bengal': 'East', 'Odisha': 'East',
        'Rajasthan': 'West', 'Gujarat': 'West', 'Maharashtra': 'West', 'Goa': 'West',
        'Madhya Pradesh': 'Central', 'Chhattisgarh': 'Central',
        'Andhra Pradesh': 'South', 'Telangana': 'South', 'Karnataka': 'South', 
        'Kerala': 'South', 'Tamil Nadu': 'South',
        'Assam': 'North East', 'Meghalaya': 'North East', 'Mizoram': 'North East', 
        'Nagaland': 'North East', 'Tripura': 'North East', 'Manipur': 'North East'
    }
    
    # 2. Regional Analysis
    # We aggregate state stats first, then map to region
    state_stats = enrol_df.groupby('state')[['age_18_greater']].sum()
    state_stats['Total_Enrol'] = enrol_df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().sum(axis=1)
    state_stats['Region'] = state_stats.index.map(region_map).fillna('Other')
    
    regional = state_stats.groupby('Region').sum()
    regional['Adult_Share_Pct'] = (regional['age_18_greater'] / regional['Total_Enrol']) * 100
    
    # 3. Volatility Analysis (CV)
    # Filter for districts with >30 days of activity
    daily = enrol_df.groupby(['state', 'district', 'date'])['age_5_17'].sum().reset_index()
    volatility = daily.groupby(['state', 'district'])['age_5_17'].agg(['mean', 'std', 'count'])
    volatility = volatility[volatility['count'] > 30].copy()
    volatility['CV_Score'] = volatility['std'] / (volatility['mean'] + 0.1) # Coefficient of Variation
    
    return regional, volatility

# Execution (Add this to your main block)
regional_stats, volatility_stats = calculate_phase2_metrics(enrol_df)

print("\n[PHASE 2] Regional Adult Enrolment Share:")
print(regional_stats['Adult_Share_Pct'].sort_values(ascending=False))

print("\n[PHASE 2] Most Volatile Districts (Likely Camps):")
print(volatility_stats.sort_values('CV_Score', ascending=False)['CV_Score'].head(5))

# ======================================================
# SNIPPET: EXPORT FULL DISTRICT DATA (FOR DASHBOARDS)
# ======================================================
def export_full_district_data(enrol, demo, bio):
    print("\n[Action] Generating Full District Master File...")
    
    # 1. Aggregate ALL Data (Outer Join to keep every district)
    e_grp = enrol.groupby(['state', 'district'])[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
    d_grp = demo.groupby(['state', 'district'])[['demo_age_5_17', 'demo_age_17_']].sum()
    b_grp = bio.groupby(['state', 'district'])[['bio_age_5_17', 'bio_age_17_']].sum()
    
    full_df = e_grp.join([d_grp, b_grp], how='outer').fillna(0)
    
    # 2. Add Totals
    full_df['Enrol_Total'] = full_df['age_0_5'] + full_df['age_5_17'] + full_df['age_18_greater']
    full_df['Update_Total'] = full_df['demo_age_5_17'] + full_df['demo_age_17_'] + full_df['bio_age_5_17'] + full_df['bio_age_17_']
    full_df['Grand_Total'] = full_df['Enrol_Total'] + full_df['Update_Total']
    
    # 3. Add Key Relations (Metrics)
    # R1: Update Efficiency (UER)
    full_df['UER_Score'] = full_df['Update_Total'] / (full_df['Enrol_Total'] + 1)
    
    # R4: Adult Entry Rate (Migration/Fraud Proxy)
    full_df['Adult_Entry_Rate'] = full_df['age_18_greater'] / (full_df['Enrol_Total'] + 1)
    
    # R3: Catch-up Index (Missed Births)
    full_df['Catch_Up_Index'] = full_df['age_5_17'] / (full_df['age_0_5'] + 1)
    
    # 4. Add Phase 2 Metrics: Volatility (CV) & Region
    # Region Map
    region_map = {
        'Jammu and Kashmir': 'North', 'Himachal Pradesh': 'North', 'Punjab': 'North', 'Uttarakhand': 'North', 'Haryana': 'North', 'Delhi': 'North', 'Uttar Pradesh': 'North',
        'Bihar': 'East', 'Jharkhand': 'East', 'West Bengal': 'East', 'Odisha': 'East',
        'Rajasthan': 'West', 'Gujarat': 'West', 'Maharashtra': 'West', 'Goa': 'West',
        'Madhya Pradesh': 'Central', 'Chhattisgarh': 'Central',
        'Andhra Pradesh': 'South', 'Telangana': 'South', 'Karnataka': 'South', 'Kerala': 'South', 'Tamil Nadu': 'South',
        'Assam': 'North East', 'Meghalaya': 'North East', 'Mizoram': 'North East', 'Nagaland': 'North East', 'Tripura': 'North East', 'Manipur': 'North East'
    }
    full_df['Region'] = full_df.index.get_level_values('state').map(region_map).fillna('Other')

    # Volatility Calculation (CV)
    if 'date' in enrol.columns:
        daily = enrol.groupby(['state', 'district', 'date'])['age_5_17'].sum().reset_index()
        vol = daily.groupby(['state', 'district'])['age_5_17'].agg(['mean', 'std'])
        full_df['CV_Volatility'] = (vol['std'] / (vol['mean'] + 0.1)).fillna(0)
    
    # 5. Export to CSV
    filename = 'aadhaar_district_analytics_full.csv'
    full_df.to_csv(filename)
    print(f"Success! Saved {len(full_df)} districts to '{filename}'")
    return full_df

# ======================================================
# SNIPPET: AGE-BUCKET BEHAVIORAL ANALYTICS
# ======================================================
def calculate_age_bucket_analytics(df):
    print("[Analysis] Calculating Age-Specific Behavioral Metrics...")
    
    # 1. The "Compliance vs Correction" Gap
    # Compare Biometric Intensity between Children and Adults
    
    # Child Bio Intensity (Mandatory Updates)
    # Formula: Child Bio Updates / Child Demo Updates
    df['R21_Child_Bio_Intensity'] = df['bio_age_5_17'] / (df['demo_age_5_17'] + 1)
    
    # Adult Bio Intensity (Voluntary/Fixes)
    # Formula: Adult Bio Updates / Adult Demo Updates
    df['R22_Adult_Bio_Intensity'] = df['bio_age_17_'] / (df['demo_age_17_'] + 1)
    
    # 2. The "Burden Shift" (Who is clogging the centers?)
    # Share of Total Transactions that are Adult Updates
    df['R23_Adult_Workload_Share'] = (df['demo_age_17_'] + df['bio_age_17_']) / df['Grand_Total']
    
    # Share of Total Transactions that are Child Enrolments (0-5)
    df['R24_Infant_Enrol_Share'] = df['age_0_5'] / df['Grand_Total']
    
    # 3. System Maturity Classification
    # If Adult Updates > 50% of work -> "Correction Phase"
    # If Child Enrol > 50% of work -> "Expansion Phase"
    conditions = [
        (df['R23_Adult_Workload_Share'] > 0.5),
        (df['R24_Infant_Enrol_Share'] > 0.2)
    ]
    choices = ['Mature (Adult Maint.)', 'Expansion (Births)']
    df['System_Phase'] = np.select(conditions, choices, default='Mixed/School Phase')
    
    return df

def plot_age_behavior(df):
    print("[Plotting] Generating Age-Behavior Comparison...")
    
    # Visualization: Child vs Adult Biometric Intensity
    # We expect Children to be High Bio, Adults to be Low Bio. Deviations are anomalies.
    
    plt.figure(figsize=(10, 6))
    
    # Filter for cleaner plot
    plot_data = df[df['Grand_Total'] > 1000].sample(frac=0.5, random_state=42) # Sample for readability
    
    sns.scatterplot(
        data=plot_data, 
        x='R22_Adult_Bio_Intensity', 
        y='R21_Child_Bio_Intensity',
        hue='Region',
        alpha=0.6
    )
    
    # Add diagonal line (Parity)
    max_val = min(plot_data['R21_Child_Bio_Intensity'].max(), 10)
    plt.plot([0, max_val], [0, max_val], 'r--', label='Equal Intensity')
    
    plt.title('Behavioral Gap: Mandatory Child Bios vs Voluntary Adult Fixes', fontsize=14)
    plt.xlabel('Adult Bio Intensity (Voluntary)', fontsize=12)
    plt.ylabel('Child Bio Intensity (Mandatory)', fontsize=12)
    plt.xlim(0, 5) # Zoom in to relevant range
    plt.ylim(0, 10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('vis_age_behavior.png')
    print("   -> Saved 'vis_age_behavior.png'")

# ==========================================
# ML MODULE: K-MEANS CLUSTERING
# ==========================================
def perform_clustering(df):
    print("[2/3] Performing ML Clustering...")
    
    # Features for clustering
    features = ['UER_Score', 'Catch_Up_Index', 'Adult_Entry_Rate', 'CV_Volatility']
    X = df[features].copy()
    
    # Handle infinite/NaN values created by div/0
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster into 4 distinct profiles
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    
    # Name the clusters based on centroids (Simplified logic)
    # We map IDs to names based on mean values
    cluster_profile = df.groupby('Cluster_ID')[features].mean()
    print("\nCluster Profiles (Centroids):")
    print(cluster_profile)
    
    return df

# ==========================================
# UPDATED EXECUTION & MERGING LOGIC
# ==========================================
def process_final_data(filename='aadhaar_district_analytics_full.csv'):
    print("[1/3] Reading and Fixing Data...")
    df = pd.read_csv(filename)
    
    # 1. Apply Cleaning (Renaming)
    df = clean_data(df) 
    
    # 2. MERGE DUPLICATES (The Critical Fix)
    # This sums up rows that now have the same 'state' and 'district' name
    print("[2/3] Merging duplicate districts...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Build aggregation dictionary
    agg_dict = {col: 'sum' for col in numeric_cols}
    if 'Region' in df.columns:
        agg_dict['Region'] = 'first' # Keep region name
    if 'CV_Volatility' in df.columns:
        agg_dict['CV_Volatility'] = 'mean' # Average the volatility, don't sum it
    
    # Perform the merge
    final_df = df.groupby(['state', 'district'], as_index=False).agg(agg_dict)
    
    # 3. Recalculate Ratios (Because Summing Ratios is wrong)
    final_df['UER_Score'] = final_df['Update_Total'] / (final_df['Enrol_Total'] + 1)
    final_df['Adult_Entry_Rate'] = final_df['age_18_greater'] / (final_df['Enrol_Total'] + 1)
    final_df['Catch_Up_Index'] = final_df['age_5_17'] / (final_df['age_0_5'] + 1)
    
    # Recalculate Age-Bucket Analytics
    final_df = calculate_age_bucket_analytics(final_df)
    
    return final_df

# ==========================================
# BASIC VISUALIZATIONS
# ==========================================
def plot_basic_visualizations(df):
    print("\n[Plotting] Generating Basic Visualizations...")
    
    # Ensure numeric columns
    cols_to_numeric = ['age_0_5', 'age_5_17', 'age_18_greater', 
                       'demo_age_5_17', 'demo_age_17_', 
                       'bio_age_5_17', 'bio_age_17_', 
                       'Enrol_Total', 'Update_Total', 'Grand_Total', 'UER_Score']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Pie Chart: Enrolment Breakdown
    total_age_0_5 = df['age_0_5'].sum()
    total_age_5_17 = df['age_5_17'].sum()
    total_age_18_plus = df['age_18_greater'].sum()
    
    plt.figure(figsize=(8, 8))
    labels = ['Age 0-5', 'Age 5-17', 'Age 18+']
    sizes = [total_age_0_5, total_age_5_17, total_age_18_plus]
    colors = ['#ff9999','#66b3ff','#99ff99']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Overall Enrolment Composition by Age Group')
    plt.savefig('vis_pie_enrolment_age.png')
    print("   -> Generated 'vis_pie_enrolment_age.png'")

    # 2. Pie Chart: Update Types
    total_demo = df['demo_age_5_17'].sum() + df['demo_age_17_'].sum()
    total_bio = df['bio_age_5_17'].sum() + df['bio_age_17_'].sum()
    
    plt.figure(figsize=(8, 8))
    labels = ['Demographic Updates', 'Biometric Updates']
    sizes = [total_demo, total_bio]
    colors = ['#ffcc99','#c2c2f0']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Overall Update Composition (Demo vs Bio)')
    plt.savefig('vis_pie_updates_type.png')
    print("   -> Generated 'vis_pie_updates_type.png'")

# ==========================================
# ADDITIONAL VISUALIZATIONS
# ==========================================
def plot_additional_visualizations(df):
    print("\n[Plotting] Generating Top 10 Charts...")
    
    # Ensure correct data types
    cols_to_numeric = ['age_0_5', 'age_5_17', 'age_18_greater', 
                       'demo_age_5_17', 'demo_age_17_', 
                       'bio_age_5_17', 'bio_age_17_']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate Totals if missing
    if 'Enrol_Total' not in df.columns:
        df['Enrol_Total'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    df['Demo_Total'] = df['demo_age_5_17'] + df['demo_age_17_']
    df['Bio_Total'] = df['bio_age_5_17'] + df['bio_age_17_']

    # --- 1. Top 10 Districts: Enrolment (Stacked) ---
    top_dist_enrol = df.sort_values('Enrol_Total', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_dist_enrol['district'], top_dist_enrol['age_0_5'], color='blue', label='Age 0-5')
    plt.bar(top_dist_enrol['district'], top_dist_enrol['age_5_17'], bottom=top_dist_enrol['age_0_5'], color='red', label='Age 5-17')
    plt.bar(top_dist_enrol['district'], top_dist_enrol['age_18_greater'], bottom=top_dist_enrol['age_0_5'] + top_dist_enrol['age_5_17'], color='yellow', label='Age 18+')
    
    plt.title('Top 10 Districts: Enrolment Breakdown', fontsize=14)
    plt.ylabel('Total Enrolment')
    plt.xlabel('District')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vis_top10_dist_enrol.png')
    print("   -> Generated 'vis_top10_dist_enrol.png'")

    # --- 2. Top 10 Districts: Demo Update (Stacked) ---
    top_dist_demo = df.sort_values('Demo_Total', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_dist_demo['district'], top_dist_demo['demo_age_5_17'], color='blue', label='Age 5-17')
    plt.bar(top_dist_demo['district'], top_dist_demo['demo_age_17_'], bottom=top_dist_demo['demo_age_5_17'], color='red', label='Age 18+')
    
    plt.title('Top 10 Districts: Demographic Updates Breakdown', fontsize=14)
    plt.ylabel('Total Demo Updates')
    plt.xlabel('District')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vis_top10_dist_demo.png')
    print("   -> Generated 'vis_top10_dist_demo.png'")

    # --- 3. Top 10 Districts: Bio Update (Stacked) ---
    top_dist_bio = df.sort_values('Bio_Total', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_dist_bio['district'], top_dist_bio['bio_age_5_17'], color='blue', label='Age 5-17')
    plt.bar(top_dist_bio['district'], top_dist_bio['bio_age_17_'], bottom=top_dist_bio['bio_age_5_17'], color='red', label='Age 18+')
    
    plt.title('Top 10 Districts: Biometric Updates Breakdown', fontsize=14)
    plt.ylabel('Total Bio Updates')
    plt.xlabel('District')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vis_top10_dist_bio.png')
    print("   -> Generated 'vis_top10_dist_bio.png'")

    # --- STATE LEVEL AGGREGATION ---
    state_df = df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater', 
                                    'demo_age_5_17', 'demo_age_17_', 
                                    'bio_age_5_17', 'bio_age_17_']].sum().reset_index()
    state_df['Enrol_Total'] = state_df['age_0_5'] + state_df['age_5_17'] + state_df['age_18_greater']
    state_df['Demo_Total'] = state_df['demo_age_5_17'] + state_df['demo_age_17_']
    state_df['Bio_Total'] = state_df['bio_age_5_17'] + state_df['bio_age_17_']

    # --- 4. Top 10 States: Enrolment (Stacked) ---
    top_state_enrol = state_df.sort_values('Enrol_Total', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_state_enrol['state'], top_state_enrol['age_0_5'], color='blue', label='Age 0-5')
    plt.bar(top_state_enrol['state'], top_state_enrol['age_5_17'], bottom=top_state_enrol['age_0_5'], color='red', label='Age 5-17')
    plt.bar(top_state_enrol['state'], top_state_enrol['age_18_greater'], bottom=top_state_enrol['age_0_5'] + top_state_enrol['age_5_17'], color='yellow', label='Age 18+')
    
    plt.title('Top 10 States: Enrolment Breakdown', fontsize=14)
    plt.ylabel('Total Enrolment')
    plt.xlabel('State')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vis_top10_state_enrol.png')
    print("   -> Generated 'vis_top10_state_enrol.png'")

    # --- 5. Top 10 States: Demo Update (Stacked) ---
    top_state_demo = state_df.sort_values('Demo_Total', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_state_demo['state'], top_state_demo['demo_age_5_17'], color='blue', label='Age 5-17')
    plt.bar(top_state_demo['state'], top_state_demo['demo_age_17_'], bottom=top_state_demo['demo_age_5_17'], color='red', label='Age 18+')
    
    plt.title('Top 10 States: Demographic Updates Breakdown', fontsize=14)
    plt.ylabel('Total Demo Updates')
    plt.xlabel('State')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vis_top10_state_demo.png')
    print("   -> Generated 'vis_top10_state_demo.png'")

    # --- 6. Top 10 States: Bio Update (Stacked) ---
    top_state_bio = state_df.sort_values('Bio_Total', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_state_bio['state'], top_state_bio['bio_age_5_17'], color='blue', label='Age 5-17')
    plt.bar(top_state_bio['state'], top_state_bio['bio_age_17_'], bottom=top_state_bio['bio_age_5_17'], color='red', label='Age 18+')
    
    plt.title('Top 10 States: Biometric Updates Breakdown', fontsize=14)
    plt.ylabel('Total Bio Updates')
    plt.xlabel('State')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vis_top10_state_bio.png')
    print("   -> Generated 'vis_top10_state_bio.png'")

if __name__ == "__main__":
    # 1. Process
    master_df = process_final_data('aadhaar_district_analytics_full.csv')
    
    # 2. Add Machine Learning
    master_df = perform_clustering(master_df)
    
    # 3. Save Final
    master_df.to_csv('aadhaar_district_analytics_ML_final.csv', index=False)
    # Also save as the cleaned file for the App to use
    master_df.to_csv('aadhaar_district_analytics_final_cleaned.csv', index=False)
    
    print("[3/3] Success! Saved corrected data to 'aadhaar_district_analytics_final_cleaned.csv'")
# ==========================================
# FINAL EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Process
    master_df = process_final_data('aadhaar_district_analytics_full.csv')
    
    # 2. Add Machine Learning
    master_df = perform_clustering(master_df)
    
    # 3. Save Final
    master_df.to_csv('aadhaar_district_analytics_ML_final.csv', index=False)
    print("[3/3] Success! Saved 'aadhaar_district_analytics_ML_final.csv'")
    
    # 4. Generate the Age Plot
    plot_age_behavior(master_df)
    plot_additional_visualizations(master_df)
    plot_basic_visualizations(master_df)

    # 5. Visualize Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=master_df[master_df['Grand_Total'] > 1000],
        x='UER_Score',
        y='Catch_Up_Index',
        hue='Cluster_ID',
        palette='viridis',
        s=100, alpha=0.7
    )
    plt.title('ML-Based District Segmentation', fontsize=14)
    plt.xlabel('Maintenance Intensity (UER)', fontsize=12)
    plt.ylabel('Growth Potential (Catch-up Index)', fontsize=12)
    plt.savefig('vis_ml_clusters.png')
    print("   -> Saved 'vis_ml_clusters.png'")
