import streamlit as st
import requests
from bs4 import BeautifulSoup
import parser
import coach
import doc_generator
import security
import eval_judge
import evaluate

# Page Configuration
st.set_page_config(
    page_title="Resume Coach AI Platform",
    page_icon="🎯",
    layout="wide"
)

def fetch_jd_from_url(url: str) -> str:
    """Fetches and extracts clean text from a job description webpage."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        clean_lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
        return "\n".join(clean_lines)
    except Exception as e:
        st.error(f"Failed to fetch job description from URL: {e}")
        return ""


# Header
st.title("🎯 Resume Coach — AI Career Assistant & MLOps Platform")
st.markdown("Production-grade resume tailoring with strict ATS gap analysis, STAR rewrites, and runtime quality evaluation.")
st.divider()

# Navigation Tabs
tab_analyzer, tab_benchmark = st.tabs(["📄 Resume Analyzer (Live User)", "🧪 Model Benchmark Lab (MLflow Evals)"])

# ==============================================================================
# TAB 1: RESUME ANALYZER
# ==============================================================================
with tab_analyzer:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📋 Step 1: Candidate Inputs")

        uploaded_file = st.file_uploader(
            "Upload Your Resume (.pdf or .docx)",
            type=["pdf", "docx"],
            help="Upload your current resume in PDF or Word format."
        )

        st.markdown("### Target Job Description")
        jd_tab1, jd_tab2 = st.tabs(["📝 Paste Text", "🔗 Paste Link (URL)"])

        jd_text = ""
        with jd_tab1:
            jd_input_text = st.text_area(
                "Paste the complete job description here:",
                height=230,
                placeholder="Paste role responsibilities, requirements, qualifications..."
            )
            if jd_input_text:
                jd_text = jd_input_text

        with jd_tab2:
            jd_url = st.text_input(
                "Paste Job Posting URL:",
                placeholder="https://company.com/careers/job-role"
            )
            if jd_url:
                with st.spinner("Fetching job description from URL..."):
                    fetched_text = fetch_jd_from_url(jd_url)
                    if fetched_text:
                        st.success("Successfully fetched job description!")
                        jd_text = fetched_text

        # Dynamic placeholder: swaps to disabled grey button on click
        analyze_btn_placeholder = st.empty()
        analyze_button = analyze_btn_placeholder.button(
            "🚀 Analyze & Optimize Resume",
            type="primary",
            use_container_width=True,
            key="btn_analyze_active"
        )

    with col_output:
        st.subheader("📊 Step 2: Coaching Insights")

        if analyze_button:
            # Swap button to greyed-out disabled state immediately
            analyze_btn_placeholder.button(
                "⏳ Analysis in Progress...",
                disabled=True,
                use_container_width=True,
                key="btn_analyze_disabled"
            )

            # 1. Quota Check
            is_allowed, current_usage, max_limit = security.check_and_increment_quota("user_session", max_daily_limit=5)
            
            if not is_allowed:
                st.error(f"🛑 Daily Usage Quota Reached ({current_usage}/{max_limit}). Quota resets automatically tomorrow morning.")
            elif not uploaded_file:
                st.warning("⚠️ Please upload your resume first.")
            elif not jd_text.strip():
                st.warning("⚠️ Please provide a target job description.")
            else:
                st.caption(f"ℹ️ Daily Quota: {current_usage}/{max_limit} analyses used today.")
                
                # 🔄 Animated Spiral Loader with Live Steps
                with st.status("🔄 Processing Resume Analysis...", expanded=True) as status_box:
                    st.write("📄 Extracting text from document...")
                    resume_text = parser.parse_resume(uploaded_file)

                    st.write("🧠 AI Coach auditing ATS fit & optimizing recent STAR bullets...")
                    results = coach.analyze_resume(resume_text, jd_text, prompt_version="v2")

                    st.write("🛡️ Running independent LLM-as-a-Judge Quality Audit...")
                    audit = eval_judge.audit_coaching_output(resume_text, jd_text, results)

                    status_box.update(label="✅ Analysis & Quality Audit Complete!", state="complete", expanded=False)

                # Display ATS Match Score
                score = results.get("match_score", 0)
                st.metric(label="🎯 ATS Match Score", value=f"{score}%")
                st.progress(score / 100)

                # Executive Summary
                st.markdown("#### 💡 Executive Summary")
                st.info(results.get("summary", "Analysis complete."))

                # Skills Gap Breakdown
                col_match, col_miss = st.columns(2)
                with col_match:
                    st.markdown("##### ✅ Matching Strengths")
                    for skill in results.get("matching_skills", []):
                        st.markdown(f"- **{skill}**")

                with col_miss:
                    st.markdown("##### ⚠️ Missing Critical Skills")
                    for skill in results.get("missing_skills", []):
                        st.markdown(f"- 🔴 {skill}")

                # STAR Rewrites Section
                st.divider()
                st.markdown("#### 🌟 Bullet Point Optimizer (STAR Framework)")
                st.caption(f"ℹ️ {results.get('rewrites_summary', '')}")

                star_rewrites = results.get("star_rewrites", [])
                if star_rewrites:
                    for idx, item in enumerate(star_rewrites, 1):
                        with st.expander(f"📌 Optimization #{idx} (Recent Work Experience)", expanded=True):
                            st.markdown(f"**❌ Original:**\n> *{item.get('original')}*")
                            st.markdown(f"**✅ STAR-Optimized:**\n> **{item.get('rewritten')}**")
                            st.caption(f"💡 **Why:** {item.get('reason')}")
                else:
                    st.write("No weak bullet points detected in recent experience.")

                # Live AI Quality Audit Badge
                st.divider()
                st.markdown("#### 🛡️ AI Quality Audit Badge (Runtime Eval)")
                col_q1, col_q2, col_q3 = st.columns(3)
                col_q1.metric("Faithfulness Score", f"{audit['faithfulness']} / 5.0")
                col_q2.metric("JD Relevance Score", f"{audit['relevance']} / 5.0")
                col_q3.metric("STAR Metric Compliance", f"{audit['star_compliance_pct']}%")
                st.success(f"Audit Verdict: **{audit['verdict']}** — {audit['notes']}")

                # Dual Download Options
                st.divider()
                st.markdown("#### 📥 Download Tailored Deliverables")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    docx_buffer = doc_generator.generate_tailored_resume(uploaded_file, resume_text, star_rewrites)
                    st.download_button(
                        label="📄 Tailored Resume (.docx)",
                        data=docx_buffer,
                        file_name="Tailored_Resume.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
                with col_dl2:
                    txt_report = doc_generator.generate_text_coaching_report(results)
                    st.download_button(
                        label="📋 Coaching Report (.txt)",
                        data=txt_report,
                        file_name="Resume_Coaching_Report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        else:
            st.info("👈 Upload your resume and paste a job description on the left, then click **Analyze & Optimize Resume**.")

# ==============================================================================
# TAB 2: MODEL BENCHMARK LAB (MLFLOW EVALS)
# ==============================================================================
with tab_benchmark:
    st.subheader("🧪 Automated Benchmark & MLOps Evaluation Lab")
    st.markdown("""
    Evaluate model quality across our curated Golden Dataset benchmarks.
    This automated test suite scores **Score Range Accuracy**, **Missing Skills Recall**, 
    and **STAR Metric Quantification**, and permanently commits the run to **MLflow**.
    """)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        sel_prompt = st.selectbox("Select Prompt Version:", ["v2", "v1"], help="v2 prioritizes recent experience with STAR rules.")
    with col_b2:
        sel_dataset = st.selectbox("Select Golden Dataset Version:", ["v2", "v1"], help="v2 contains 20 comprehensive scenarios.")

    # Dynamic placeholder for benchmark button
    bench_btn_placeholder = st.empty()
    run_bench_btn = bench_btn_placeholder.button(
        "🚀 Run Live Benchmark Against Golden Dataset",
        type="primary",
        key="btn_bench_active"
    )

    if run_bench_btn:
        # Swap button to greyed-out disabled state immediately
        bench_btn_placeholder.button(
            "⏳ Benchmark in Progress...",
            disabled=True,
            type="primary",
            key="btn_bench_disabled"
        )

        # 🔄 Animated Spiral Loader for Benchmark
        with st.status(f"🧪 Executing Benchmark ({sel_dataset} with {sel_prompt})...", expanded=True) as bench_status:
            st.write("📂 Loading Golden Benchmark Dataset...")
            st.write("⚡ Running multi-case evaluation against Groq LLM...")
            metrics = evaluate.run_evaluation(prompt_version=sel_prompt, dataset_version=sel_dataset)
            bench_status.update(label="✅ Benchmark Completed & Logged to MLflow!", state="complete", expanded=False)

        st.success("✅ Logged to MLflow Experiment: `Resume_Coach_Evals`!")
        st.divider()

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Score Range Accuracy", f"{metrics.get('score_accuracy_pct', 0)}%")
        m_col2.metric("Missing Skills Recall", f"{metrics.get('missing_skills_recall_pct', 0)}%")
        m_col3.metric("STAR Metric Rate", f"{metrics.get('star_quantification_pct', 0)}%")
        m_col4.metric("Average Latency", f"{metrics.get('avg_latency_seconds', 0)}s")

        st.info(f"Tested across {metrics.get('test_cases_count', 0)} benchmark scenarios. View detailed traces via `mlflow ui --port 5000`.")