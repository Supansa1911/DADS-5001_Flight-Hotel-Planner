import pandas as pd
import duckdb

##def prepare(file_path):
def prepare(sheet_name, file_path="flight_hotel_clean.xlsx.xlsx"):
    
    try:
        print(f"🔍 Loading {sheet_name} sheet from path: {file_path}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        ## ระบุ sheet_name
        ##df = pd.read_excel(f"{file_path}", sheet_name="Hotel")
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        return df
    except Exception as e:
        print("❌ Error loading file:", e)
        return pd.DataFrame()  # ส่ง dataframe ว่างคืนกลับ ถ้าโหลดไม่ได้



