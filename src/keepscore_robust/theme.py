from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f7f5;
            color: #111111;
        }
        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        .ks-page-title {
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 0.35rem 0;
        }
        .ks-nav-wrap {
            display:flex;
            justify-content:space-between;
            gap:1rem;
            align-items:center;
            padding: 0.45rem 0 0.95rem 0;
            border-bottom:1px solid #e3e3e0;
            margin-bottom: 1rem;
            flex-wrap:wrap;
        }
        .ks-brand {
            font-weight: 800;
            letter-spacing: .04em;
            font-size: 1.12rem;
        }
        .ks-links {
            color: #555555;
            font-weight: 500;
            font-size: 0.95rem;
        }
        .ks-hero {
            background: linear-gradient(135deg, #ffffff 0%, #ffffff 68%, #f1f1ef 100%);
            border: 1px solid #e3e3e0;
            border-radius: 24px;
            padding: 1.5rem 1.7rem;
            margin-bottom: 0.85rem;
        }
        .ks-hero h1 {
            margin: 0.25rem 0 0.45rem 0;
            font-size: 2.5rem;
            line-height: 1.05;
        }
        .ks-hero p {
            font-size: 1rem;
            max-width: 60rem;
            color: #333333;
            margin-bottom: 0;
        }
        .ks-eyebrow {
            color:#c1121f;
            font-weight:700;
            letter-spacing:.08em;
            font-size:.8rem;
        }
        .ks-help-strip {
            display:grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.8rem;
            margin: 0 0 1rem 0;
        }
        .ks-help-strip > div {
            background: #ffffff;
            border: 1px solid #e5e5e2;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            font-size: 0.93rem;
        }
        div[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e3e3e0;
        }
        div[data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid #e7e7e4;
            border-radius: 16px;
            padding: 0.2rem 0.35rem;
        }
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            border-radius: 999px;
            border: 1px solid #d9d9d5;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid #e3e3e0;
            padding: 0 18px;
        }
        .stTabs [aria-selected="true"] {
            background: #111111 !important;
            color: #ffffff !important;
        }
        .ks-card {
            border: 1px solid rgba(0,0,0,0.12);
            border-radius: 18px;
            padding: 1.25rem 1.25rem 1rem 1.25rem;
            background: white;
            min-height: 290px;
            margin-bottom: 0.75rem;
        }
        .ks-card-title {
            font-size: 1.15rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.75rem;
        }
        .ks-card-sub {
            font-size: 0.95rem;
            color: rgba(0,0,0,0.58);
            line-height: 1.5;
            margin-bottom: 1.35rem;
        }
        .ks-stat-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1.1rem;
            margin-bottom: 1.25rem;
        }
        .ks-stat-label {
            font-size: 0.9rem;
            color: rgba(0,0,0,0.58);
            margin-bottom: 0.25rem;
        }
        .ks-stat-value {
            font-size: 1.9rem;
            font-weight: 600;
            line-height: 1.05;
            color: #111111;
        }
        .ks-card-reasons {
            font-size: 0.95rem;
            line-height: 1.65;
            color: #1f1f1f;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2.3rem;
        }
        @media (max-width: 900px) {
            .ks-help-strip {
                grid-template-columns: 1fr;
            }
            .ks-hero h1 {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
