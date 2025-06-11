import streamlit as st
import pandas as pd
from modules.data_prep import prepare
import duckdb
from pymongo import MongoClient
import plotly.express as px
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rc('font', family='Tahoma')  # หรือ 'TH Sarabun New' ถ้ามี
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Tahoma'
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pymongo
from datetime import datetime
import uuid


# ตั้งค่าหน้า
st.set_page_config(page_title="Trip Dashboard", page_icon="🧳")


#st.set_page_config(page_title="Trip Analysis App", layout="wide")
st.title("🛫 Flight & Hotel Planner")

page = st.sidebar.selectbox("เลือกหน้าที่ต้องการดู:", ["📈 Data Prep & Model Hotel", "🧠 Data Prep & Model Flight", "🏨 Hotel Analysis", "✈️ Flight Analysis","🏠 หน้าแรก","🎯 Flight & Hotel Recommendation","🎫 Booking System"])

if page == "🏠 หน้าแรก":

    st.markdown("## 🔍 ยินดีต้อนรับสู่ระบบการวางแผนเที่ยวบินและโรงแรมที่ดีที่สุดสำหรับคุณ")
    st.write("เริ่มต้นวางแผนการท่องเที่ยวได้ง่ายๆ ครบจบในที่เดียว ได้ที่นี่!!")
    #st.write("เลือกเมนูด้านซ้ายเพื่อดูการวิเคราะห์ข้อมูลจาก Excel โดยใช้ SQL (DuckDB) และ NoSQL (MongoDB) พร้อมกราฟ Plotly")

    # API key
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Gemini setup
    genai.configure(api_key="AIzaSyBIdDIyQwTn1fRiJLYwoXxSFRwM0NrGOAE")  
    model = genai.GenerativeModel("gemini-1.5-flash-latest")


    # ========== LOAD DATA ==========
    @st.cache_data
    def load_data():
        xls = pd.ExcelFile("HotelFlight_28 Sep.xlsx")
        hotels = xls.parse('Hotel')
        flights = xls.parse('Flight')
        return hotels, flights

    hotels, flights = load_data()

    # ========== STREAMLIT APP UI ==========
    st.title("🧳 คำถามสำหรับการวางแผนการเดินทางของคุณ")

    # ========== SESSION STATE ==========
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ========== CLEAR HISTORY BUTTON ==========
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []

    # ========== USER INPUT ==========
    user_question = st.text_area("💬 Ask your question", placeholder="e.g., Which hotel under 5000 THB has the highest rating?")

    if st.button("🔍 Ask Gemini"):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Gemini is thinking..."):

                # Convert tables to markdown
                hotel_text = hotels.to_markdown(index=False)
                flight_text = flights.to_markdown(index=False)

                prompt = f"""
    You are a helpful AI travel assistant. The user has provided hotel and flight data in table format.
    Your job is to answer questions based ONLY on the data.

    ## Hotel Table:
    {hotel_text}

    ## Flight Table:
    {flight_text}

    Now answer this user question:
    {user_question}

    Be brief, friendly, and data-focused. If you cannot answer based on the data, say so.
    """

                try:
                    response = model.generate_content(prompt)
                    answer = response.text

                    # Save to chat history
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Gemini", answer))

                except Exception as e:
                    st.error(f"Gemini error: {e}")

    # ========== DISPLAY CHAT HISTORY ==========
    st.markdown("## 💬 Chat History")
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧍 You**: {message}")
        else:
            st.markdown(f"**🤖 Gemini**: {message}")
    
  
       

elif page == "🏨 Hotel Analysis":
    df_hotel = prepare("Hotel")
    df_hotel.columns = df_hotel.columns.str.strip()  # ล้างช่องว่าง

    st.header("🏨 Hotel Analysis")
    st.markdown("""
    ตารางข้อมูลโรงแรม โดยมีชื่อโรงแรม, ราคา, Rating ที่โรงแรมได้รับ, จำนวน Review             
    ข้อมูลจาก Trip.com โดยข้อมูลเข้าพักวันที่ 19 ตุลาคม 2567 (1 คืน)
    """)
    st.dataframe(df_hotel)


    if df_hotel.empty:
        st.error("ไม่สามารถโหลดข้อมูล Hotel ได้")
    else:
        # สร้าง connection duckdb และลงทะเบียน DataFrame
        con = duckdb.connect()
        con.register('df', df_hotel)

        st.subheader("🪶 DuckDB SQL Analysis: Top 5 Hotels with Highest Average Price")
        st.write("จากการใช้ DuckDB : SQL Analysis ในการหา 5 อันดับแรกที่มีราคาเฉลี่ยโรงแรมสูงที่สุด")
        query_result = con.execute("""
            SELECT hotel_name, AVG("price(บาท)") as avg_price
            FROM df
            GROUP BY hotel_name
            ORDER BY avg_price DESC
            LIMIT 5
        """).fetchdf()
        st.dataframe(query_result)

        # เชื่อมต่อ MongoDB
        st.subheader("🍃 MongoDB NoSQL Analysis")
        st.write("จากการใช้ MongoDB : NO SQL Analysis ในการหา 5 อันดับแรกพบมากที่สุด")
        client = MongoClient("mongodb://localhost:27017")
        db = client["trip_db"]
        collection = db["hotel_data"]

        # เช็คและ insert ข้อมูลถ้ายังไม่มี
        if collection.count_documents({}) == 0:
            collection.insert_many(df_hotel.to_dict("records"))

        top_hotels = collection.aggregate([
            {"$group": {"_id": "$hotel_name"}},
            {"$sort": {"total_bookings": -1}},
            {"$limit": 5}
        ])
        top_hotels_df = pd.DataFrame(top_hotels)
        st.dataframe(top_hotels_df)

        # Hotel Rating Trend
        df_hotel.columns = df_hotel.columns.str.strip()  # ลบช่องว่างหัวท้าย
        df_hotel = df_hotel.rename(columns={
            'Hotel name': 'hotel_name',
            'Rating (เต็ม 5)': 'rating'
        })



elif page == "✈️ Flight Analysis":
    df_flight = prepare("Flight (2)")
    st.header("✈️ Flight Analysis")
    st.markdown("""
    ตารางข้อมูลเที่ยวบิน โดยมีชื่อสายการบิน, การต่อเครื่อง, เวลาออกเดินทาง, เวลาถึง, ราคา, ราคาที่มีส่วนลด และจำนวนชั่วโมง            
    ข้อมูลจาก Trip.com โดยข้อมูลเที่ยวบินขาไปวันที่ 19 ตุลาคม 2567 และขากลับวันที่ 26 ตุลาคม 2567
    """)
    st.dataframe(df_flight)
    #st.footer("ข้อมูลจาก Trip.com โดย flight check in: Oct 19,2024, check out: Oct 26,2024 ")

    con = duckdb.connect()
    con.register('df', df_flight)
    st.subheader("🪶 DuckDB SQL Analysis: Top 5 Airlines with Highest Price")
    st.write("จากการใช้ DuckDB : SQL Analysis ในการหา5อันดับแรกของสายการบินที่มีราคาเที่ยวบินสูงที่สุด")
    query_result = con.execute("""
            SELECT DISTINCT "สายการบิน", "ราคาเต็ม(บาท)" 
            From df  
            ORDER BY "ราคาเต็ม(บาท)" DESC
            LIMIT 5
        """).fetchdf()
    st.dataframe(query_result)

    if df_flight.empty:
        st.error("ไม่สามารถโหลดข้อมูล Flight ได้")
    else:
        numeric_cols = df_flight.select_dtypes(include='number').columns.tolist()

        st.divider()
        st.subheader("🔸 Scatter Plot")
        st.write("สามารถเลือกดูTrend กราฟ จากข้อมูลตามที่ User ต้องการ")
        x_axis = st.selectbox("เลือกแกน X:", numeric_cols, index=0)
        y_axis = st.selectbox("เลือกแกน Y:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        fig = px.scatter(df_flight, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}")
        st.plotly_chart(fig)

        st.divider()
        st.subheader("📆 Box Plot")
        st.write("สามารถเลือกดูTrend กราฟ จากข้อมูลตามที่ User ต้องการ")
        box_col = st.selectbox("เลือกคอลัมน์:", numeric_cols)
        fig = px.box(df_flight, y=box_col, title=f"Box Plot: {box_col}")
        st.plotly_chart(fig)

        st.divider()
        st.subheader("📊 Distribution Plot (Histogram)")
        st.write("สามารถเลือกดูTrend กราฟ จากข้อมูลตามที่ User ต้องการ")
        hist_col = st.selectbox("เลือกคอลัมน์ (hist):", numeric_cols, key="hist")
        fig = px.histogram(df_flight, x=hist_col, nbins=20, title=f"Distribution: {hist_col}")
        st.plotly_chart(fig)

        st.divider()
        st.subheader("🪧 Bar Plot")
        st.write("สามารถเลือกดูTrend กราฟ จากข้อมูลตามที่ User ต้องการ")
        cat_cols = df_flight.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            cat_col = st.selectbox("เลือกคอลัมน์หมวดหมู่:", cat_cols)
            bar_data = df_flight[cat_col].value_counts().reset_index()
            bar_data.columns = [cat_col, 'count']
            fig = px.bar(bar_data, x=cat_col, y='count', title=f"Bar Plot: {cat_col}")
            st.plotly_chart(fig)
        else:
            st.warning("ไม่พบคอลัมน์ประเภทหมวดหมู่")

# --------------------------
# 🎯 Flight & Hotel Recommendation
# --------------------------
elif page == "🎯 Flight & Hotel Recommendation":

    st.title("✈️ ระบบแนะนำเที่ยวบิน (โดยอิงจากกลุ่มที่คล้ายกัน)")

        # 📄 Prepare df_no_arrive again for this section only
    df = pd.read_excel("Flight_for PCA.xlsx", sheet_name="Flight")
    df_no_arrive = df.copy()
    le = LabelEncoder()
    #df_encoded["Airlines"] = le.fit_transform(df_encoded["Airlines"])
    df_no_arrive["Airlines"] = le.fit_transform(df_no_arrive["Airlines"])
    features_no_arrive = ["Airlines", "Depart", "Flight hours", "Price"]
    X_no_arrive = df_no_arrive[features_no_arrive]
    X_scaled_no_arrive = StandardScaler().fit_transform(X_no_arrive)

        # PCA + KMeans clustering
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled_no_arrive)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_no_arrive["Cluster"] = kmeans.fit_predict(X_pca)

        # 📜 UI 䃢ลือกสายการบิน
    airlines_list_display = df["Airlines"].unique()
    selected_airline = st.selectbox("เลือกสายการบินที่คุณสนใจ", sorted(airlines_list_display))

        # 💸 䃠ลือราคาที่ต้องการ
    min_price = int(df["Price"].min())
    max_price = int(df["Price"].max())
    price_range = st.slider("เลือกราคาที่คุณต้องการ (บาท)", min_value=min_price, max_value=max_price, value=(min_price, max_price))

    if selected_airline:
        selected_df = df[df["Airlines"] == selected_airline]
        if selected_df.empty:
            st.warning("ไม่พบสายการบินนี้ในระบบ")
        else:
            encoded = le.transform([selected_airline])[0]
            selected_cluster = df_no_arrive[df_no_arrive["Airlines"] == encoded]["Cluster"].values[0]

            st.success(f"สายการบิน '{selected_airline}' อยู่ในกลุ่มที่ {selected_cluster}")

            filtered = df_no_arrive[(df_no_arrive["Cluster"] == selected_cluster) &
                                        (df_no_arrive["Price"] >= price_range[0]) &
                                        (df_no_arrive["Price"] <= price_range[1])]

            st.markdown(f"### ✈️ เที่ยวบินที่อยู่ในกลุ่มเดียวกัน และช่วงราคา {price_range[0]:,} - {price_range[1]:,} บาท")
            st.dataframe(filtered[["Airlines", "Depart", "Flight hours", "Price"]])

    st.divider()
    # Pages/hotel_recommend.py
    st.title("🏨 ระบบแนะนำโรงแรม (จากกลุ่มที่คล้ายกัน)")

    # โหลดข้อมูลจากไฟล์โรงแรม
    df = pd.read_excel("HotelFlight_28 Sep.xlsx", sheet_name="Hotel")
    df.columns = df.columns.str.strip()  # เคลียร์ช่องว่าง

    # เลือก features ที่จะใช้ (ใช้เฉพาะ Price เท่านั้น)
    selected_features = ["Hotel name", "Price(บาท)", "Review (คน)","Rating (เต็ม 5)"]
    df = df[selected_features].dropna()

    # เตรียมข้อมูลสำหรับ KMeans
    X = df[["Price(บาท)"]]
    X_scaled = StandardScaler().fit_transform(X)

    # สร้างกลุ่มด้วย KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # UI: เลือกโรงแรม
    hotel_list = df["Hotel name"].unique()
    selected_hotel = st.selectbox("เลือกโรงแรมที่คุณสนใจ", sorted(hotel_list))

    # UI: เลือกราคา
    min_price = int(df["Price(บาท)"].min())
    max_price = int(df["Price(บาท)"].max())
    price_range = st.slider("เลือกราคาที่ต้องการ (บาท)", min_value=min_price, max_value=max_price, value=(min_price, max_price))

    if selected_hotel:
        selected_cluster = df[df["Hotel name"] == selected_hotel]["Cluster"].values[0]
        st.success(f"โรงแรม '{selected_hotel}' อยู่ในกลุ่มที่ {selected_cluster}")

        filtered = df[(df["Cluster"] == selected_cluster) &
                      (df["Price(บาท)"] >= price_range[0]) &
                      (df["Price(บาท)"] <= price_range[1])]

        st.markdown(f"### 🏨 โรงแรมในกลุ่มเดียวกัน และช่วงราคา {price_range[0]:,} - {price_range[1]:,} บาท")
        st.dataframe(filtered[["Hotel name", "Price(บาท)", "Review (คน)","Rating (เต็ม 5)"]])



    

elif page == "🧠 Data Prep & Model Flight":
    # 📌 Load data
    df = pd.read_excel("Flight_for PCA.xlsx", sheet_name="Flight")
    df_encoded = df.copy()
    le = LabelEncoder()
    df_encoded["Airlines"] = le.fit_transform(df_encoded["Airlines"])

    # 🎯 Feature Selection
    features = ["Airlines", "Depart", "Flight hours", "Price"]
    X = df_encoded[features]
    X_scaled = StandardScaler().fit_transform(X)

    # 🎯 PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_encoded[["PC1", "PC2"]] = X_pca

    # 🎯 KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_encoded["Cluster"] = kmeans.fit_predict(X_pca)

    # 📊 Cluster Summary
    summary = df_encoded.groupby("Cluster").agg({
        "Price": ["mean", "min", "max", "count"],
        "Flight hours": "mean",
        "Depart": "mean"
    })
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.reset_index(inplace=True)
    airline_examples = []
    for c in summary["Cluster"]:
        sample = df[df_encoded["Cluster"] == c]["Airlines"].unique()[:3]
        airline_examples.append(', '.join(sample))
    summary["Airline_Examples"] = airline_examples

    # ✅ หน้า Navigation
    #st.set_page_config(page_title="Flight App", layout="wide")
    #page = st.sidebar.radio("เลือกหน้า", ["🧠 อธิบายการวิเคราะห์", "✈️ ระบบแนะนำเที่ยวบิน"])
    
    st.divider()
    # =====================================
    # 🧠 หน้าอธิบายการวิเคราะห์
    # =====================================
    #st.title ("🧠 อธิบายการวิเคราะห์")
    st.title("🧠 การประยุกต์ใช้ PCA เข้ากับ K-Mean")
    st.markdown("""
    ทำการประยุกต์ใช้ความรู้จากวิชา DADS 6003 โดยใช้เทคนิคการ **ลดมิติด้วย PCA** มาช่วยในทำ **K-Means** เพื่อหากลุ่มเที่ยวบินที่มีลักษณะใกล้เคียงกันได้อย่างมีประสิทธิภาพมากขึ้น
    """)

    st.divider()
    st.subheader("โมเดลที่ทำวิเคราะห์เบื้องต้นด้วย K-Means (ใช้ทุกตัวแปร)")
    st.markdown("""
    ในวิชา DADS 5002 ได้ทำการวิเคราะห์เบื้องต้นโดยใช้ K-Means กับข้อมูลเที่ยวบิน โดยใช้ตัวแปรทั้งหมดที่มีและไม่มีการทำ PCA ก่อน""")
    features_all = ["Airlines", "Depart", "Arrive", "Flight hours", "Price"]
    df_all = df.copy()
    df_all["Airlines"] = le.fit_transform(df_all["Airlines"])
    X_all = df_all[features_all]
    X_scaled_all = StandardScaler().fit_transform(X_all)

        # KMeans เบื้องต้น
    kmeans_initial = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_all["Cluster"] = kmeans_initial.fit_predict(X_scaled_all)

        # Visualize
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_all, x="Depart", y="Price", hue="Cluster", palette="tab10", ax=ax)
    ax.set_title(" Cluster Visualization (Depart vs Price)")
    st.pyplot(fig)

    st.markdown("จากการใช้ตัวแปรทั้งหมด และเลือกกลุ่ม 5 กลุ่ม จะเห็นว่ามีการกระจายของข้อมูลที่ไม่ชัดเจนมากนัก ยังมีการทับซ้อนกันระหว่างกลุ่มต่างๆ")



    st.subheader("ประยุกต์ใช้ PCA เพื่อดูความสำคัญของตัวแปร")

    pca_all = PCA(n_components=len(features_all))
    X_pca_all = pca_all.fit_transform(X_scaled_all)

        # ตาราง Loadings
    loadings_all = pd.DataFrame(
        pca_all.components_.T,
        columns=[f"PC{i+1}" for i in range(len(features_all))],
        index=features_all
    )

    explained_var_all = pd.Series(
        pca_all.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(len(features_all))]
    )

        # แสดง Scree plot
    fig, ax = plt.subplots()
    sns.barplot(x=explained_var_all.index, y=explained_var_all.values, ax=ax)
    ax.set_title("Explained Variance by PC (ใช้ทุกตัวแปร)")
    st.pyplot(fig)

        # แสดงตาราง
    st.dataframe(loadings_all.style.format("{:.4f}"))

    st.markdown("### 🧾 สรุปการวิเคราะห์ตัวแปรจาก PCA")

    st.markdown("""
    <style>
    .feature-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 16px;
    }
    .feature-table th, .feature-table td {
        border: 1px solid #ccc;
        padding: 8px 12px;
        text-align: left;
    }
    .feature-table th {
        background-color: #f0f2f6;
        color: #333;
    }
    .feature-table td:first-child {
        font-weight: bold;
    }
    .highlight-green {
        color: green;
        font-weight: bold;
    }
    .highlight-orange {
        color: orange;
        font-weight: bold;
    }
    </style>

    <table class="feature-table">
        <tr>
            <th>Feature</th>
            <th>Analysis</th>
        </tr>
        <tr>
            <td>Airlines</td>
            <td>มีน้ำหนักสูงใน PC1, PC2, PC3 → <span class="highlight-green">เก็บไว้</span></td>
        </tr>
        <tr>
            <td>Depart</td>
            <td>โดดเด่นใน PC2 (-0.7565) และมีบทบาทใน PC5 → <span class="highlight-green">เก็บไว้</span></td>
        </tr>
        <tr>
            <td>Arrive</td>
            <td>เด่นเฉพาะ PC3 (0.8841) แต่ต่ำใน PC1, PC2, PC5 → <span class="highlight-orange">ใช้เฉพาะกรณีสนใจเวลา Arrive</span></td>
        </tr>
        <tr>
            <td>Flight hours</td>
            <td>มีผลใน PC1 และ PC4 → <span class="highlight-green">เก็บไว้</span></td>
        </tr>
        <tr>
            <td>Price</td>
            <td>สำคัญใน PC1 และ PC5 → <span class="highlight-green">สำคัญแน่นอน</span></td>
    </tr>
    </table>
    """, unsafe_allow_html=True)


    st.divider()
    st.subheader("ตัดตัวแปร 'Arrive' ออก แล้วทำ PCA")

        # ตัด Arrive ออก
    features_no_arrive = ["Airlines", "Depart", "Flight hours", "Price"]
    df_no_arrive = df.copy()
    df_no_arrive["Airlines"] = le.fit_transform(df_no_arrive["Airlines"])
    X_no_arrive = df_no_arrive[features_no_arrive]
    X_scaled_no_arrive = StandardScaler().fit_transform(X_no_arrive)

        # ทำ PCA ใหม่ทั้งหมด (ไม่กำหนด n_components)
    pca_ref = PCA()
    X_pca_ref = pca_ref.fit_transform(X_scaled_no_arrive)
    explained_var_ref = pd.Series(
        pca_ref.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(len(features_no_arrive))]
    )

        # ตาราง Loadings หลังตัด Arrive
    loadings_refined = pd.DataFrame(
        pca_ref.components_.T,
        columns=explained_var_ref.index,
        index=features_no_arrive
    )

        # Scree plot เต็มแบบเหมือนรอบแรก
    fig, ax = plt.subplots()
    sns.barplot(x=explained_var_ref.index, y=explained_var_ref.values, ax=ax)
    ax.set_title("Explained Variance by PC (หลังตัดตัวแปร Arrive)")
    ax.set_ylabel("Proportion of Variance Explained")
    ax.set_xlabel("Principal Component")
    st.pyplot(fig)

        # ตารางโหลดดิ้ง
    st.markdown("""
    ### 📋 ตาราง Loadings ของตัวแปรในแต่ละ Principal Component (หลังตัดตัวแปร Arrive)
    """)
    st.dataframe(loadings_refined.style.format("{:.4f}"))

        # คำอธิบายประกอบ
    st.markdown("""
    จากกราฟ **Explained Variance by PC** (หลังตัดตัวแปร `Arrive`) เราจะเห็นว่า:

    - **PC1** อธิบายข้อมูลได้สูงที่สุด (~41%)
    - **PC2** รองลงมา (~27%)
    - โดยรวม **PC1 + PC2** รวมกันได้มากกว่า 68% ซึ่งเพียงพอต่อการลดมิติเหลือ 2 แกนเพื่อวิเคราะห์ต่อไป

    > ✨ เราจึงเลือกใช้เฉพาะ 2 แกนหลักในการทำ Elbow และจัดกลุ่มด้วย K-Means
    """)
        # ทำ PCA เหลือ 2 มิติ
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled_no_arrive)
    explained_var_pca2d = pd.Series(
        pca_2d.explained_variance_ratio_,
        index=["PC1", "PC2"]
    )

        # Visualize PCA 2D (ไม่แยกกลุ่มก่อน)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], ax=ax)
    ax.set_title("การกระจายของข้อมูลหลังทำ PCA (2 มิติ)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)


        # แสดง Loadings ของแต่ละฟีเจอร์หลังตัด Arrive
    loadings_refined = pd.DataFrame(
        pca_2d.components_.T,
        columns=["PC1", "PC2"],
        index=features_no_arrive
    )
    

    st.subheader("ใช้เทคนิค Elbow Method กับข้อมูลหลังทำ PCA (2 มิติ)")

        # Elbow Method บนข้อมูลที่ผ่าน PCA
    inertia_pca = []
    K_range = range(1, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca_2d)
        inertia_pca.append(km.inertia_)

    fig, ax = plt.subplots()
    sns.lineplot(x=K_range, y=inertia_pca, marker="o", ax=ax)
    ax.set_title("Elbow Method บนข้อมูล PCA 2 มิติ")
    ax.set_xlabel("จำนวนคลัสเตอร์ (k)")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    st.markdown("""
    จากการใช้ elbow method เราจะได้ค่า **k=3** หรือ 3 กลุ่มนั่นเอง  
    ซึ่งเป็นจำนวนกลุ่มที่เหมาะสมในการนำไปใช้กับ K-Means ต่อไป
    """)

    st.subheader("ทำ K-Means บน PCA (2 มิติ)")

        # ✅ ใช้ k = 3 ตาม Elbow
    optimal_k = 3
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_no_arrive["Cluster"] = kmeans_final.fit_predict(X_pca_2d)

        # 🎨 แสดงผลการจัดกลุ่ม
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=df_no_arrive["Cluster"], palette="tab10", ax=ax)
    ax.set_title(f"K-Means Clustering บน PCA (2 มิติ) | k = {optimal_k}")
    st.pyplot(fig)

    st.markdown(f"""
    เราทำการจัดกลุ่ม K-Means โดยใช้ **k = 3** ซึ่งได้จาก Elbow Method  
    ผลลัพธ์แสดงให้เห็นการแยกกลุ่มที่ชัดเจน โดยอิงจากข้อมูลที่ลดเหลือ 2 มิติผ่าน PCA
    """)

    st.subheader("📊 สรุปข้อมูลของแต่ละกลุ่ม (Cluster = 3)")

    summary_final = df_no_arrive.copy()
    summary_table = summary_final.groupby("Cluster").agg({
        "Price": ["mean", "min", "max", "count"],
        "Flight hours": "mean",
        "Depart": "mean"
    })
    summary_table.columns = ['_'.join(col) for col in summary_table.columns]
    summary_table.reset_index(inplace=True)

    st.dataframe(summary_table)

    st.subheader("📌 สรุปกลุ่มจากการทำ Clustering (หลังตัดตัวแปร Arrive)")

        # จัดรูปแบบข้อความแสดงผลให้สวยงาม
    cluster_descriptions = [
        {
            "title": "Cluster 0",
            "airlines": "All Nippon Airways, Hong Kong Airlines, Air Canada, All Nippon AirwaysCodeshare flight",
            "depart": "14.42 น.",
            "hours": "21.53 ชม.",
            "price": "฿57,182 (min: ฿31,495, max: ฿119,705)",
            "count": 22
        },
        {
            "title": "Cluster 1",
            "airlines": "Thai Airways, Air Canada, Korean Air, KLM Royal Dutch Airlines",
            "depart": "17.59 น.",
            "hours": "32.28 ชม.",
            "price": "฿105,043 (min: ฿60,420, max: ฿211,065)",
            "count": 9
        },
        {
            "title": "Cluster 2",
            "airlines": "China Airlines, EVA Air, Japan Airlines, Air Canada",
            "depart": "4.61 น.",
            "hours": "26.19 ชม.",
            "price": "฿50,758 (min: ฿38,275, max: ฿78,920)",
            "count": 10
        }
    ]

    for c in cluster_descriptions:
        st.markdown(f"""
        <div style='border:1px solid #ccc; border-radius:10px; padding:16px; margin-bottom:10px; background-color:#f9f9f9;'>
            <h4>📌 {c['title']}</h4>
            <ul>
                <li><b>ตัวอย่างสายการบิน:</b> {c['airlines']}</li>
                <li><b>เวลาออกเดินทางเฉลี่ย:</b> {c['depart']}</li>
                <li><b>ชั่วโมงบินเฉลี่ย:</b> {c['hours']}</li>
                <li><b>ราคาเฉลี่ย:</b> {c['price']}</li>
                <li><b>จำนวนเที่ยวบิน:</b> {c['count']} เที่ยวบิน</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)



elif page == "📈 Data Prep & Model Hotel":
    st.title("📊 Correlation: Hotel Insights")
    st.markdown("""
    ทำการประยุกต์ใช้ความรู้จากวิชา DADS 6001 โดยการใช้ Correlation มาช่วยในการทำ Regression เพื่อทำนายค่าโรงแรม หากมีค่า **Rating** และจำนวน **Review** เป็นตัวแปรหลัก
    """)

    st.divider()

        # ========== LOAD DATA ==========
    @st.cache_data
    def load_data():
        xls = pd.ExcelFile("HotelFlight_28 Sep.xlsx")
        df_hotel = pd.read_excel(xls, sheet_name="Hotel")
        return df_hotel


    df_hotel = load_data()
    # ----------------------
    # Hotel Correlation
    # ----------------------
    st.subheader("📈 Correlation Matrix - Hotel")

    hotel_corr = df_hotel.select_dtypes(include=[np.number]).corr()
    fig_hotel, ax_hotel = plt.subplots()
    sns.heatmap(hotel_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_hotel)
    ax_hotel.set_title("Hotel Dataset Correlation")
    st.pyplot(fig_hotel)

    # 🔍 Pearson r และ p-value (หลัง dropna)
    df_corr = df_hotel[['Price(บาท)', 'Rating (เต็ม 5)', 'Review (คน)']].dropna()
    r1, p1 = pearsonr(df_corr['Price(บาท)'], df_corr['Rating (เต็ม 5)'])
    r2, p2 = pearsonr(df_corr['Price(บาท)'], df_corr['Review (คน)'])

    st.markdown(f"""
    ### 🔍 Pearson Correlation Analysis (Hotel)
    จากการวิเคราะห์ค่าสัมประสิทธิ์สหสัมพันธ์ (Correlation Analysis) สำหรับข้อมูลโรงแรม Hotel Data พบว่า:

    - ระหว่าง `'Price(บาท)'` และ `'Rating (เต็ม 5)'`  
      → ค่าสัมประสิทธิ์สหสัมพันธ์ **r = {r1:.4f}**, และค่า **p-value = {p1:.4f}**  
      → **สรุปได้ว่า:** {'ไม่มีความสัมพันธ์กันอย่างมีนัยสำคัญ' if p1 >= 0.05 else 'มีความสัมพันธ์กันอย่างมีนัยสำคัญ'}

    - ระหว่าง `'Price(บาท)'` และ `'Review (คน)'`  
      → ค่าสัมประสิทธิ์สหสัมพันธ์ **r = {r2:.4f}**, และค่า **p-value = {p2:.4f}**  
      → **สรุปได้ว่า:** {'ไม่มีความสัมพันธ์กันอย่างมีนัยสำคัญ' if p2 >= 0.05 else 'มีความสัมพันธ์กันอย่างมีนัยสำคัญ'}
    """)



    st.divider()
    st.title("📊 Hotel Price Regression Analysis")

    # โหลดข้อมูล
    df = pd.read_excel("HotelFlight_28 Sep.xlsx", sheet_name="Hotel")

    # ทำความสะอาด
    df = df.dropna(subset=["Rating (เต็ม 5)", "Review (คน)", "Price(บาท)"])
    df = df.rename(columns={
        "Rating (เต็ม 5)": "Rating",
        "Review (คน)": "Reviews",
        "Price(บาท)": "Price"
    })

    # เตรียม X และ y
    X1 = df[["Rating", "Reviews"]]  # Model 1: Rating + Reviews
    X2 = df[["Rating"]]             # Model 2: Rating Only
    y = df["Price"]

    # ฟังก์ชันประเมิน Linear Regression
    def evaluate_model(X, y):
        model = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
        return r2, rmse, cv_r2

    # ผล Linear Regression 2 แบบ
    results_lr = {
        "Model 1: Rating + Reviews": evaluate_model(X1, y),
        "Model 2: Rating Only": evaluate_model(X2, y)
    }
    results_lr_df = pd.DataFrame(results_lr, index=["R²", "RMSE", "CV Mean R²"]).T.round(3)

    # แสดงผลตาราง Linear Regression
    st.subheader("🔍 Linear Regression เปรียบเทียบแบบ 2 โมเดล")
    st.dataframe(results_lr_df)

    # สรุปผล Linear Regression
    st.markdown("""
    **🔎 สรุป Linear Regression**
    - การใช้ฟีเจอร์ Rating + Reviews ให้ผลดีกว่า Rating อย่างเดียวเล็กน้อย
    - แต่ R² และค่า Cross-validation R² ยังคงติดลบ แสดงว่าอธิบายราคาห้องได้ไม่ดี
    """)

    # ประเมินหลายโมเดล
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    model_results = []
    for name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X1, y, cv=5, scoring="r2")

        model_results.append({
          "Model": name,
            "rmse": rmse,
            "r2_score": r2,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores)
        })

    model_results_df = pd.DataFrame(model_results).set_index("Model").round(4)

    # แสดงผลตารางโมเดลหลายแบบ
    st.subheader("📊 เปรียบเทียบโมเดลหลายแบบ (ใช้ Rating + Reviews)")
    st.dataframe(model_results_df)

    # สรุปผลเปรียบเทียบโมเดล
    st.markdown("""
    ### 📌 สรุปผลการทดลองหลายโมเดล

    - **Gradient Boosting** และ **Linear Regression** มี RMSE ใกล้เคียงกันที่สุด
    - โมเดลทั้งหมดมี R² และค่า Cross-validation R² ติดลบ → โมเดลยังไม่สามารถอธิบายราคาห้องได้ดี
    - ความผันผวนของโมเดลอย่าง Random Forest และ Boosting (CV Std สูง) บ่งชี้ถึงการ overfit

    **➡️ ข้อเสนอแนะ:**
    - เพิ่มตัวแปรอื่น เช่น ประเภทโรงแรม, ที่ตั้ง, ฤดูกาล, หรือรีวิวเชิงข้อความ
    - ทดลอง Feature Engineering เพิ่มเติม เช่น Interaction Terms หรือ Polynomial Features
    """)




#------------------
#  Bookink System 
#------------------

elif page == "🎫 Booking System":
    # MongoDB connection
    @st.cache_resource
    def init_connection():
        try:
            # Replace with your MongoDB connection string
            client = MongoClient('mongodb://localhost:27017/')
            return client
        except Exception as e:
            st.error(f"Failed to connect to MongoDB: {e}")
            return None

    # Fetch flights data from MongoDB
    @st.cache_data
    def load_flights():
        client = init_connection()
        if client is None:
            return []
        
        try:
            db = client['booking_system']  # Change this to your database name
            collection = db['flights']      # Change this to your flights collection name
            flights = list(collection.find())
            return flights
        except Exception as e:
            st.error(f"Error fetching flights: {e}")
            return []

    # Fetch hotels data from MongoDB
    @st.cache_data
    def load_hotels():
        client = init_connection()
        if client is None:
            return []
        
        try:
            db = client['booking_system']  # Change this to your database name
            collection = db['hotels']       # Change this to your hotels collection name
            hotels = list(collection.find())
            return hotels
        except Exception as e:
            st.error(f"Error fetching hotels: {e}")
            return []

    # Save booking to MongoDB
    def save_booking(booking_data):
        client = init_connection()
        if client is None:
            return False
        
        try:
            db = client['booking_system']
            bookings_collection = db['booking_system']  # Collection to store bookings
            
            # Add timestamp and booking ID
            booking_data['booking_id'] = str(uuid.uuid4())
            booking_data['booking_date'] = datetime.now()
            booking_data['status'] = 'confirmed'
            
            result = bookings_collection.insert_one(booking_data)
            return result.inserted_id is not None
        except Exception as e:
            st.error(f"Error saving booking: {e}")
            return False

    def format_option_display(item, price_field, name_field):
        """Helper function to format display options for selectbox"""
        name = item.get(name_field, 'Unknown')
        price = item.get(price_field, 0)
        
        if isinstance(price, (int, float)):
            formatted_price = f"฿{price:,.0f}"
        else:
            formatted_price = f"฿{price}"
        
        return f"{name} - {formatted_price}"

    def main():
        st.title("🎫 เลือกตั๋วเที่ยวบินและที่พักของคุณ")
        st.write("ในการวางแผนเที่ยวบินและโรงแรมที่ดีที่สุดของคุณ เราจะดำเนินการจองตั๋วเที่ยวบินและที่พักที่คุณเข้ามาบันทึก เพื่อทริปที่ดีที่สุดของคุณ พร้อมพบกับสิทธิพิเศษหากคุณเลือกเที่ยวบินและที่พักกับเรา รับส่วนลดพิเศษ 10% สำหรับที่พัก")
        st.divider()
        st.markdown("### เลือกเที่ยวบินและที่พักของคุณ ได้เลย !")
        # Load data
        flights = load_flights()
        hotels = load_hotels()
        
        if not flights and not hotels:
            st.warning("ไม่พบแผนการเดินทาง. โปรดตรวจสอบการเชื่อม่อฐานข้อมูล.")
            return
        
        # Create tabs for different booking types
        tab1, tab2, tab3 = st.tabs(["✈️ เที่ยวบิน", "🏨 ที่พัก", "🎫 เที่ยวบิน + ที่พัก"])
        
        with tab1:
            handle_flight_booking(flights)
        
        with tab2:
            handle_hotel_booking(hotels)
        
        with tab3:
            handle_combined_booking(flights, hotels)

    def handle_flight_booking(flights):
        if not flights:
            st.warning("ไม่พบเที่ยวบิน.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("เลือกเที่ยวบิน")
            
            # Create flight options
            flight_options = []
            for f in flights:
                display = format_option_display(f, 'ราคาเต็ม(บาท)', 'สายการบิน')
                flight_options.append(display)
            
            selected_flight_display = st.selectbox(
                "เลือกเที่ยวบินที่ต้องการ:",
                flight_options,
                key="flight_select"
            )
            
            selected_index = flight_options.index(selected_flight_display)
            selected_flight = flights[selected_index]
            
            # Flight booking form
            with st.form("ฟอร์มการจองเที่ยวบิน"):
                st.subheader("ข้อมูลการจองเที่ยวบิน")
                
                passenger_name = st.text_input("ชื่อผู้โดยสาร")
                passenger_email = st.text_input("อีเมล")
                passenger_phone = st.text_input("เบอร์โทร")
                travel_date = st.date_input("วันออกเดินทาง")
                passengers = st.number_input("จำนวนผู้โดยสาร", min_value=1, max_value=10, value=1)
                special_requests = st.text_area("เพิ่มเติม (Optional)")
                
                submitted = st.form_submit_button("ยืนยัน")
                
                if submitted:
                    if passenger_name and passenger_email:
                        # Prepare booking data
                        booking_data = {
                            'booking_type': 'flight',
                            'passenger_name': passenger_name,
                            'passenger_email': passenger_email,
                            'passenger_phone': passenger_phone,
                            'travel_date': travel_date.isoformat(),
                            'number_of_passengers': passengers,
                            'special_requests': special_requests,
                            'flight_details': selected_flight,
                            'total_price': selected_flight.get('ราคาเต็ม(บาท)', 0) * passengers
                        }
                        
                        # Remove MongoDB _id from flight_details
                        if '_id' in booking_data['flight_details']:
                            del booking_data['flight_details']['_id']
                        
                        # Save to MongoDB
                        if save_booking(booking_data):
                            st.success(f"จองตั๋วสำเร็จแล้ว ! {passenger_name}!")
                            st.balloons()
                        else:
                            st.error("การจองเกิดข้อผิดพลาด. โปรดลองใหม่อีกครั้ง.")
                    else:
                        st.error("โปรดกรอกรายละเอียดให้ครบถ้วน (ชื่อและอีเมล)")
        
        with col2:
            show_flight_details(selected_flight)

    def handle_hotel_booking(hotels):
        if not hotels:
            st.warning("ไม่พบโรงแรม.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("เลือกโรงแรม")
            
            # Create hotel options - adjust field names based on your hotel data structure
            hotel_options = []
            for h in hotels:
                # Modify these field names to match your hotel data structure
                display = format_option_display(h, 'Price(บาท)','Hotel name')
                hotel_options.append(display)
            
            selected_hotel_display = st.selectbox(
                "เลือกโรงแรมที่ต้องการ:",
                hotel_options,
                key="hotel_select"
            )
            
            selected_index = hotel_options.index(selected_hotel_display)
            selected_hotel = hotels[selected_index]
            
            # Hotel booking form
            with st.form("ฟอร์มการองที่พัก"):
                st.subheader("ข้อมูลการจองที่พัก")
                
                guest_name = st.text_input("ชื่อผู้เข้าพัก")
                guest_email = st.text_input("อีเมล")
                guest_phone = st.text_input("เบอร์โทร")
                check_in_date = st.date_input("Check-in Date")
                check_out_date = st.date_input("Check-out Date")
                rooms = st.number_input("จำนวนห้อง", min_value=1, max_value=10, value=1)
                guests = st.number_input("จำนวนผู้เข้าพัก", min_value=1, max_value=20, value=2)
                special_requests = st.text_area("เพิ่มเติม (Optional)")
                
                submitted = st.form_submit_button("ยืนยัน")
                
                if submitted:
                    if guest_name and guest_email and check_out_date > check_in_date:
                        # Calculate nights and total price
                        nights = (check_out_date - check_in_date).days
                        room_price = selected_hotel.get('ราคาต่อคืน', 0)
                        total_price = room_price * nights * rooms
                        
                        # Prepare booking data
                        booking_data = {
                            'booking_type': 'hotel',
                            'guest_name': guest_name,
                            'guest_email': guest_email,
                            'guest_phone': guest_phone,
                            'check_in_date': check_in_date.isoformat(),
                            'check_out_date': check_out_date.isoformat(),
                            'nights': nights,
                            'number_of_rooms': rooms,
                            'number_of_guests': guests,
                            'special_requests': special_requests,
                            'hotel_details': selected_hotel,
                            'total_price': total_price
                        }
                        
                        # Remove MongoDB _id from hotel_details
                        if '_id' in booking_data['hotel_details']:
                            del booking_data['hotel_details']['_id']
                        
                        # Save to MongoDB
                        if save_booking(booking_data):
                            st.success(f"Hotel booked successfully for {guest_name}!")
                            st.balloons()
                            st.info(f"Total Amount: ฿{total_price:,.0f} for {nights} nights")
                        else:
                            st.error("Failed to save booking. Please try again.")
                    else:
                        st.error("Please fill in all required fields and ensure check-out is after check-in")
        
        with col2:
            show_hotel_details(selected_hotel)

    def handle_combined_booking(flights, hotels):
        if not flights or not hotels:
            st.warning("Both flight and hotel data are required for combined booking.")
            return
        
        st.subheader("เลือกเที่ยวบิน + ที่พัก")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Select Flight:**")
            flight_options = [format_option_display(f, 'ราคาเต็ม(บาท)', 'สายการบิน') for f in flights]
            selected_flight_display = st.selectbox("Flight:", flight_options, key="combo_flight")
            selected_flight = flights[flight_options.index(selected_flight_display)]
        
        with col2:
            st.write("**Select Hotel:**")
            hotel_options = [format_option_display(h, 'Price(บาท)', 'Hotel name') for h in hotels]
            selected_hotel_display = st.selectbox("Hotel:", hotel_options, key="combo_hotel")
            selected_hotel = hotels[hotel_options.index(selected_hotel_display)]
        
        # Combined booking form
        with st.form("เที่ยวบิน + ที่พัก"):
            st.subheader("ข้อมูลการจองเที่วบิน + ที่พัก")
            
            col3, col4 = st.columns(2)
            with col3:
                customer_name = st.text_input("ชื่อผู้โดยสาร")
                customer_email = st.text_input("อีเมล")
                customer_phone = st.text_input("เบอร์โทร")
                travel_date = st.date_input("Travel Date")
            
            with col4:
                check_in_date = st.date_input("Hotel Check-in")
                check_out_date = st.date_input("Hotel Check-out")
                passengers = st.number_input("ผู้เข้าพัก", min_value=1, max_value=10, value=2)
                rooms = st.number_input("จำนวนห้อง", min_value=1, max_value=10, value=1)
            
            special_requests = st.text_area("เพิ่มเติม (Optional)")

            submitted = st.form_submit_button("ยืนยัน")
        
            if submitted:
                if customer_name and customer_email and check_out_date > check_in_date:
                    # Calculate costs
                    nights = (check_out_date - check_in_date).days
                    flight_cost = selected_flight.get('ราคาเต็ม(บาท)', 0) * passengers
                    hotel_cost = selected_hotel.get('ราคาต่อคืน', 0) * nights * rooms
                    total_cost = flight_cost + hotel_cost
                    discount = total_cost * 0.1  # 10% package discount
                    final_cost = total_cost - discount
                    
                    # Prepare booking data
                    booking_data = {
                        'booking_type': 'package',
                        'customer_name': customer_name,
                        'customer_email': customer_email,
                        'customer_phone': customer_phone,
                        'travel_date': travel_date.isoformat(),
                        'check_in_date': check_in_date.isoformat(),
                        'check_out_date': check_out_date.isoformat(),
                        'nights': nights,
                        'passengers': passengers,
                        'rooms': rooms,
                        'special_requests': special_requests,
                        'flight_details': selected_flight,
                        'hotel_details': selected_hotel,
                        'flight_cost': flight_cost,
                        'hotel_cost': hotel_cost,
                        'discount': discount,
                        'total_cost': final_cost
                    }
                    
                    # Remove MongoDB _ids
                    if '_id' in booking_data['flight_details']:
                        del booking_data['flight_details']['_id']
                    if '_id' in booking_data['hotel_details']:
                        del booking_data['hotel_details']['_id']
                    
                    # Save to MongoDB
                    if save_booking(booking_data):
                        st.success(f"คุณ {customer_name}ได้ทำการจองสำเร็จ!")
                        st.balloons()
                        
                        # Show cost breakdown
                        st.subheader("รายละเอียด:")
                        st.write(f"เที่ยวบิน: ฿{flight_cost:,.0f}")
                        st.write(f"ที่พัก: ฿{hotel_cost:,.0f}")
                        st.write(f"รวม: ฿{total_cost:,.0f}")
                        st.write(f"ส่วนลด (10%): -฿{discount:,.0f}")
                        st.write(f"**สุทธิ: ฿{final_cost:,.0f}**")
                    else:
                        st.error("การจองผิดพลาด โปรดลองใหม่อีกครั้ง.")
                else:
                    st.error("โปรดกรอกรายละเอียดให้ครนบถ้วน")

    def show_flight_details(selected_flight):
        st.subheader("เลือกเที่ยวบินที่ต้องการ")
        if selected_flight:
            st.info("ข้อมูลเที่ยวบิน")
            for key, value in selected_flight.items():
                if key != '_id':
                    if key == 'ราคาเต็ม(บาท)' and isinstance(value, (int, float)):
                        st.write(f"**{key}:** ฿{value:,.0f}")
                    else:
                        st.write(f"**{key}:** {value}")

    def show_hotel_details(selected_hotel):
        st.subheader("เลือกที่พักที่ต้องการ")
        if selected_hotel:
            st.info("ข้อมูลที่พัก")
            for key, value in selected_hotel.items():
                if key != '_id':
                    if key == 'ราคาต่อคืน' and isinstance(value, (int, float)):
                        st.write(f"**{key}:** ฿{value:,.0f}")
                    else:
                        st.write(f"**{key}:** {value}")

    # View bookings section
    def show_bookings():
        st.divider()
        st.subheader("📋 สถิติการจอง")
        
        client = init_connection()
        if client is None:
            return
        
        try:
            db = client['booking_system']
            bookings_collection = db['booking_system']
            
            # Get recent bookings
            recent_bookings = list(bookings_collection.find().sort('booking_date', -1).limit(10))
            
            if recent_bookings:
                # Create DataFrame for display
                bookings_df = pd.DataFrame(recent_bookings)
                
                # Clean up display
                if '_id' in bookings_df.columns:
                    bookings_df = bookings_df.drop('_id', axis=1)
                
                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("การจองทั้งหมด", len(recent_bookings))
                with col2:
                    flight_bookings = len([b for b in recent_bookings if b.get('booking_type') == 'flight'])
                    st.metric("จำนวนจองเที่ยวบิน", flight_bookings)
                with col3:
                    hotel_bookings = len([b for b in recent_bookings if b.get('booking_type') == 'hotel'])
                    st.metric("จำนวนจองที่พัก", hotel_bookings)

            else:
                st.info("ไม่พบการจอง.")
                
        except Exception as e:
            st.error(f"Error fetching bookings: {e}")

    if __name__ == "__main__":
        main()
        show_bookings()

