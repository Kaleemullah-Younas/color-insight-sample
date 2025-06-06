import streamlit as st
import google.generativeai as genai
from vision import analyze_image
import io
import re
from fpdf import FPDF

# ——— Gemini Configuration ———
def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

# ——— Strip Markdown for PDF ———
def strip_markdown(md: str) -> str:
    text = md
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*',   r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)',  r'\1', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

# ——— Sanitize Unicode → Latin-1 Safe ———
def sanitize_for_pdf(txt: str) -> str:
    # map common smart‐quotes, dashes, ellipses → ASCII
    tbl = {
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-',
        '\u2026': '...',
    }
    for uni, repl in tbl.items():
        txt = txt.replace(uni, repl)
    # drop any remaining non-Latin-1
    return txt.encode('latin-1', 'ignore').decode('latin-1')

# ——— PDF Generation ———
def create_pdf_report(user_data, analysis_text):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(33, 33, 33)
    pdf.cell(0, 15, sanitize_for_pdf("Color Analysis Report"), ln=True, align='C')
    pdf.ln(5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(10)

    # Personal Information Table
    pdf.set_font("Arial", 'B', 16)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_for_pdf("Personal Information"), ln=True, fill=True)
    pdf.ln(4)

    pdf.set_font("Arial", '', 12)
    col_w = [40, 140]
    for label, key in [
        ("Name", "name"), ("Age", "age"), ("Email", "email"),
        ("Gender", "gender"), ("Height", "height"),
        ("Body Shape", "body_shape"), ("Color Preference", "color_preference"),
        ("Accessories", "accessories")
    ]:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(col_w[0], 8, sanitize_for_pdf(f"{label}:"), border=0)
        pdf.set_font("Arial", '', 12)
        value = str(user_data.get(key, ""))
        pdf.multi_cell(col_w[1], 8, sanitize_for_pdf(value), border=0)
    pdf.ln(8)

    # Analysis Sections
    pdf.set_font("Arial", 'B', 16)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, sanitize_for_pdf("Analysis Report"), ln=True, fill=True)
    pdf.ln(4)

    pdf.set_font("Arial", '', 12)
    heading_re = re.compile(r'^\d+\.\s+')
    for line in analysis_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        safe_line = sanitize_for_pdf(line)
        if heading_re.match(safe_line):
            pdf.ln(4)
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 8, safe_line)
            pdf.set_font("Arial", '', 12)
        else:
            pdf.multi_cell(0, 6, safe_line)
    pdf.ln(10)

    # Footer
    pdf.alias_nb_pages()
    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, sanitize_for_pdf(f"Page {pdf.page_no()}/{{nb}}"), align='C')

    return pdf.output(dest='S').encode('latin-1')

# ——— Streamlit UI ———
st.title("Color Insights Analysis")

with st.form("user_data_form"):
    name   = st.text_input("Name")
    email  = st.text_input("Email")
    age    = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height_unit = st.selectbox("Height Unit", ["inches", "cm"])
    height = st.number_input(f"Height ({height_unit})")
    body_shape = st.selectbox("Body Shape", ["Rectangle","Inverted Rectangle","Trapezoid","Triangle","Oval"])
    color_preference = st.selectbox("Colors You Normally Wear",
        ["Pastel and soft colors","Bright and vibrant colors","Earthy tones","Neutral colors"])
    accessories = st.selectbox("Accessories You Wear",
        ["Bright: Shiny, clear/crystal","Light: Feminine, pearl","Mute: Matte, rose gold",
         "Dark: Fancy, big, bold","I do not wear accessories"])
    image_source = st.radio("Image Source", ["Upload Photo", "Capture Live"])
    if image_source == "Upload Photo":
        image_file = st.file_uploader("Upload your photo", type=['jpg','jpeg','png'])
    else:
        image_file = st.camera_input("Take a photo")
    gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")
    submit_button  = st.form_submit_button("Analyze")

if submit_button and image_file and gemini_api_key:
    image_bytes  = image_file.getvalue()
    feature_list, labels = analyze_image(image_bytes)

    # Print to console
    print("🚀 Face feature list:", feature_list)
    print("🔖 Detected labels:", labels)

    user_data = {
        "name": name, "email": email, "age": age, "gender": gender,
        "height": f"{height} {height_unit}", "body_shape": body_shape,
        "color_preference": color_preference, "accessories": accessories,
        "facial_features": feature_list, "detected_labels": labels
    }

    # Generate analysis
    model = configure_gemini(gemini_api_key)
    prompt = (
        "You are a professional color & style consultant. "
        "Given the user data below, generate a detailed recommendation report using these headings:\n\n"
        "1. Personal Color Season Analysis\n"
        "2. Best Color Combinations\n"
        "3. Style Recommendations Based on Body Shape\n"
        "4. Accessory Suggestions\n"
        "5. Makeup Color Recommendations\n\n"
        "Use a formal expert tone, omit any casual introductions or assumptions, "
        "and provide only straight-to-the-point guidance under each heading.\n\n"
        "User Data:\n"
        f"{user_data}\n"
    )
    response = model.generate_content(prompt)
    analysis_result = response.text

    # Display on UI
    st.subheader("Your Color Analysis Report")
    st.markdown(analysis_result)

    # Clean & PDF
    clean_text = strip_markdown(analysis_result)
    pdf_bytes  = create_pdf_report(user_data, clean_text)

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="color_analysis_report.pdf",
        mime="application/pdf"
    )