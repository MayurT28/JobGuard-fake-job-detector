import streamlit as st
import time
from predict import combined_verdict, explain

st.set_page_config(
    page_title="JobGuard — Fake Job Detector",
    page_icon="🛡",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
footer { display: none; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }

html, body, .stApp {
    background-color: #f8f7f4 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

section[data-testid="stMain"] > div {
    padding-top: 28px;
    padding-left: 16px !important;
    padding-right: 16px !important;
}

h1, h2, h3, p, label, div, span {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stTextArea textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    background: #ffffff !important;
    border: 1px solid #e5e4e0 !important;
    border-radius: 10px !important;
    color: #1a1917 !important;
    line-height: 1.7 !important;
    padding: 14px !important;
}

.stTextArea textarea:focus {
    border-color: #1a1917 !important;
    box-shadow: none !important;
}

.stTextArea label { display: none !important; }

.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    background: #1a1917 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 24px !important;
    width: 100% !important;
    letter-spacing: -0.01em !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

.stSpinner > div { border-top-color: #1a1917 !important; }
</style>
""", unsafe_allow_html=True)

# ── NAV ──
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding-bottom:18px;border-bottom:1px solid #e5e4e0;margin-bottom:32px;
     flex-wrap:wrap;gap:10px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:30px;height:30px;background:#1a1917;border-radius:7px;
         display:flex;align-items:center;justify-content:center">
      <span style="color:white;font-size:13px">🛡</span>
    </div>
    <span style="font-size:15px;font-weight:700;color:#1a1917;
         letter-spacing:-0.02em">JobGuard</span>
  </div>
  <span style="font-size:12px;color:#1a6b3c;font-weight:500">● Model live</span>
</div>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div style="margin-bottom:28px">
  <p style="font-size:11px;font-weight:600;letter-spacing:0.1em;
     text-transform:uppercase;color:#9b9a95;margin-bottom:10px">
    AI-powered · India-focused · BERT + Llama 3
  </p>
  <h1 style="font-size:clamp(26px,5vw,34px);font-weight:700;
     letter-spacing:-0.03em;line-height:1.15;color:#1a1917;margin-bottom:10px">
    Is this job posting real?<br>Find out in seconds.
  </h1>
  <p style="font-size:14px;color:#5c5b56;line-height:1.65">
    Trained on 17,880 postings. 93% recall on fake jobs. 98% accuracy on Indian data.
  </p>
</div>
""", unsafe_allow_html=True)

# ── INPUT ──
st.markdown("""
<div style="background:#fff;border:1px solid #e5e4e0;border-radius:12px;
     overflow:hidden;margin-bottom:10px">
  <div style="padding:13px 16px;border-bottom:1px solid #e5e4e0">
    <span style="font-size:13px;font-weight:600;color:#1a1917">
      Paste job description
    </span>
  </div>
""", unsafe_allow_html=True)

job_text = st.text_area(
    label="job",
    label_visibility="collapsed",
    height=180,
    placeholder="Paste the full job posting here — title, company, responsibilities, requirements, contact details, salary...",
    key="job_input"
)

st.markdown("</div>", unsafe_allow_html=True)

analyse = st.button("Analyse job posting", use_container_width=True)

if analyse:
    if not job_text.strip():
        st.warning("Please paste a job description first.")
    else:
        with st.spinner("Analysing..."):
            final_label, final_conf, strong_signals, weak_signals = combined_verdict(job_text)
            explanation = explain(job_text, final_label, final_conf)

        # ── RISK-LEVEL COLOR SYSTEM ──

        if final_label == "FAKE" or final_conf < 0.60:
            # HIGH RISK (RED)
            v_color = "#c0392b"
            v_bg = "#fdf2f1"
            v_border = "#f0c4c0"
            tag_bg = "#f0c4c0"
            tag_text = "High risk posting"
            v_title = "Multiple caution indicators detected"

        elif final_conf < 0.80:
            # MEDIUM RISK (YELLOW)
            v_color = "#b7791f"
            v_bg = "#fffaf0"
            v_border = "#f5d9a8"
            tag_bg = "#f5d9a8"
            tag_text = "Needs verification"
            v_title = "Some caution indicators present"

        else:
            # LOW RISK (GREEN)
            v_color = "#1a6b3c"
            v_bg = "#f0f9f4"
            v_border = "#b8e0ca"
            tag_bg = "#b8e0ca"
            tag_text = "Appears legitimate"
            v_title = "No strong fraud indicators found"
        conf_pct = f"{final_conf*100:.0f}%"

        # ── VERDICT ──
        st.markdown(f"""
        <div style="background:{v_bg};border:1px solid {v_border};
             border-radius:14px;padding:22px 20px;margin:18px 0 12px">
          <div style="display:flex;align-items:center;justify-content:space-between;
               flex-wrap:wrap;gap:10px;margin-bottom:12px">
            <div>
              <span style="display:inline-block;background:{tag_bg};color:{v_color};
                   font-size:11px;font-weight:700;letter-spacing:0.07em;
                   text-transform:uppercase;padding:3px 10px;border-radius:99px;
                   margin-bottom:8px">{tag_text}</span>
              <div style="font-size:17px;font-weight:700;color:{v_color};
                   letter-spacing:-0.02em">{v_title}</div>
            </div>
            <div style="text-align:center">
              <div style="font-size:10px;text-transform:uppercase;
                   letter-spacing:0.08em;color:#9b9a95;margin-bottom:2px;
                   font-weight:500">Confidence</div>
              <div style="font-size:30px;font-weight:700;
                   color:{v_color};letter-spacing:-0.02em">{conf_pct}</div>
            </div>
          </div>
          <div style="font-size:13px;color:#5c5b56;line-height:1.75;
               padding-top:12px;border-top:1px solid {v_border}">
            {explanation}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── SIGNALS ──
        # ── STRONG SIGNALS (REAL RED FLAGS) ──
        if strong_signals:

            sig_items = ""

            for s in strong_signals:
                sig_items += f"""
                <div style="display:flex;gap:10px;align-items:flex-start;
                    padding:9px 0;border-bottom:1px solid #f5d9a8;
                    font-size:13px;color:#5c5b56;line-height:1.5">
                  <div style="width:18px;height:18px;border-radius:50%;
                      background:#fef8f0;border:1px solid #f5d9a8;
                      display:flex;align-items:center;justify-content:center;
                      font-size:9px;font-weight:700;color:#92500a;
                      flex-shrink:0;margin-top:2px">!</div>
                  <span>{s}</span>
                </div>"""

            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e5e4e0;
                border-radius:12px;overflow:hidden;margin-bottom:12px">
              <div style="display:flex;align-items:center;justify-content:space-between;
                  padding:13px 16px;border-bottom:1px solid #e5e4e0">
                <span style="font-size:13px;font-weight:600;color:#1a1917">
                  Red flags detected
                </span>
                <span style="background:#fef8f0;color:#92500a;
                    border:1px solid #f5d9a8;font-size:11px;font-weight:700;
                    padding:2px 9px;border-radius:99px">
                  {len(strong_signals)} found
                </span>
              </div>
              <div style="padding:4px 16px 8px">{sig_items}</div>
            </div>
            """, unsafe_allow_html=True)


        # ── WEAK SIGNALS (VERIFICATION SUGGESTIONS) ──
        if weak_signals:

            weak_items = ""

            for s in weak_signals:
                weak_items += f"• {s}<br>"
                

            st.markdown(f"""
            <div style="background:#f7f7f7;border:1px solid #e5e4e0;
                border-radius:12px;padding:12px 16px;margin-bottom:12px">
              <span style="font-size:13px;font-weight:600;color:#1a1917">
                Additional verification suggested
              </span>
              <div style="margin-top:6px;font-size:13px;color:#5c5b56">
              {weak_items}
              </div>
            </div>
            """, unsafe_allow_html=True)


        # ── NO SIGNALS ──
        if not strong_signals and not weak_signals:

            if final_conf < 0.60:

                fallback_suggestions = [
                    "Verify recruiter contact method before applying",
                    "Check company profile on LinkedIn or official website",
                    "Avoid sharing personal documents before interview confirmation"
                ]

                weak_items = ""

                for s in fallback_suggestions:
                    weak_items += f"• {s}<br>"

                st.markdown(f"""
                <div style="background:#f7f7f7;border:1px solid #e5e4e0;
                    border-radius:12px;padding:12px 16px;margin-bottom:12px">
                  <span style="font-size:13px;font-weight:600;color:#1a1917">
                    Additional verification suggested
                  </span>
                  <div style="margin-top:6px;font-size:13px;color:#5c5b56">
                  {weak_items}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div style="background:#f0f9f4;border:1px solid #b8e0ca;
                    border-radius:12px;padding:14px 18px;margin-bottom:12px;
                    font-size:13px;font-weight:500;color:#1a6b3c">
                  ✓ No suspicious indicators detected
                </div>
                """, unsafe_allow_html=True)

        # ── DISCLAIMER ──
        st.markdown("""
        <p style="font-size:11px;color:#9b9a95;margin-top:6px;line-height:1.6">
          AI can make mistakes. Always verify the company independently
          before sharing personal details or paying any fee.
        </p>
        """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div style="margin-top:52px;padding-top:18px;border-top:1px solid #e5e4e0;
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
  <span style="font-size:11px;color:#9b9a95">
    Built by Mayur Tonge · BERT + Llama 3 · Trained on 17,880 postings
  </span>
  <span style="font-size:11px;color:#9b9a95">v1.0 · India-focused</span>
</div>
""", unsafe_allow_html=True)