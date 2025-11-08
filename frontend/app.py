# app.py
import streamlit as st
import requests

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬")
st.title("💬 Sentiment Analysis App (FastAPI Backend)")
st.write("Nhập vào đoạn văn để phân tích cảm xúc (Positive / Neutral / Negative).")

user_input = st.text_area("Nhập nội dung:", placeholder="Ví dụ: Sản phẩm rất tốt!", height=150)

if st.button("Phân tích cảm xúc"):
    if user_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung trước khi phân tích.")
    else:
        response = requests.post(f"{BASE_URL}/predict", json={"text": user_input})
        if response.status_code == 200:
            result = response.json()
            label = result["label"].lower()
            score = result["score"]

            if label == "tích cực":
                color = "green"
                emoji = "😊"
            elif label == "tiêu cực":
                color = "red"
                emoji = "😞"
            else:
                color = "gray"
                emoji = "😐"

            st.markdown(f"**Kết quả:** <span style='color:{color};font-size:22px'>{label.capitalize()}</span>", unsafe_allow_html=True)
            st.progress(score)
            st.caption(f"Độ tin cậy: {score:.2%}")
        else:
            st.error("Không thể kết nối tới API backend.")

if "records" not in st.session_state:
    st.session_state.records = []
    st.session_state.offset = 0
    st.session_state.limit = 50
    st.session_state.has_more = True

def load_more():
    try:
        res = requests.get(f"{BASE_URL}/records", params={"limit": st.session_state.limit, "offset": st.session_state.offset})
        if res.status_code == 200:
            new = res.json()
            st.session_state.records.extend(new)
            st.session_state.offset += len(new)
            # If we got fewer records than requested, there are no more records
            if len(new) < st.session_state.limit:
                st.session_state.has_more = False
        else:
            st.error("Không thể lấy dữ liệu lịch sử từ backend.")
    except Exception:
        st.error("Lỗi khi kết nối tới backend để lấy dữ liệu lịch sử.")

st.markdown("---")
st.header("Recent records")

# Initial load
if len(st.session_state.records) == 0:
    load_more()

# Display all records
for rec in st.session_state.records:
    st.markdown(f"**[{rec['timestamp']}]**  - *{rec['sentiment']}*")
    st.write(rec["text"])
    st.markdown("---")

# Show Load more button only if there might be more records
if st.session_state.has_more and st.button("⬇️ Load more", key="load_more_button"):
    load_more()
    st.rerun()