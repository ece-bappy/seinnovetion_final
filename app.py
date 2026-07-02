import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core import (
    build_or_load_index,
    extract_text_from_upload,
    grade_submission,
    load_rubric,
)

st.set_page_config(page_title="AI Assignment Grader", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def get_index():
    return build_or_load_index()


@st.cache_data(show_spinner=False)
def get_rubric_text():
    return load_rubric()


def render_questions_section():
    st.subheader("Questions")
    with st.form("questions_form"):
        uploaded_questions = st.file_uploader(
            "Upload a questions PDF/TXT/MD",
            type=["pdf", "txt", "md"],
            key="questions_upload",
        )
        st.session_state.setdefault("questions_input", "")
        questions_text = st.text_area(
            "Questions text",
            value=st.session_state.get("questions_input", ""),
            height=220,
            placeholder="Paste the assignment questions here or upload a PDF above.",
            key="questions_input",
        )
        submitted = st.form_submit_button("Save questions", use_container_width=True)

        if submitted:
            if uploaded_questions is not None:
                extracted = extract_text_from_upload(uploaded_questions)
                if extracted.strip():
                    st.session_state["saved_questions_text"] = extracted
                    st.session_state["questions_source"] = uploaded_questions.name
                    st.success("Questions saved from the uploaded file.")
                else:
                    st.warning("The uploaded file did not contain extractable text.")
            elif questions_text.strip():
                st.session_state["saved_questions_text"] = questions_text
                st.session_state["questions_source"] = "manual"
                st.success("Questions saved.")
            else:
                st.error("Please provide questions before continuing.")

    if st.session_state.get("saved_questions_text"):
        with st.expander("Saved questions", expanded=True):
            st.write(st.session_state["saved_questions_text"])


def render_single_grading(index, rubric_text):
    st.subheader("Single submission grading")
    answer_file = st.file_uploader(
        "Upload a student answer PDF/TXT/MD",
        type=["pdf", "txt", "md"],
        key="single_answer_upload",
    )

    if st.button("Evaluate", type="primary"):
        if not st.session_state.get("saved_questions_text"):
            st.error("Please save the questions first.")
            return
        if answer_file is None:
            st.error("Please upload a student answer file.")
            return

        with st.spinner("Evaluating the student answer..."):
            answer_text = extract_text_from_upload(answer_file)
            result = grade_submission(
                answer_text,
                rubric_text,
                index,
                questions_text=st.session_state["saved_questions_text"],
            )
            result["source_file"] = answer_file.name
            render_results(result)


def render_bulk_grading(index, rubric_text):
    st.subheader("Bulk grading")
    st.caption("Upload multiple student answer files and grade them sequentially.")
    answer_files = st.file_uploader(
        "Upload one or more student answer files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="bulk_answer_upload",
    )
    if st.button("Evaluate all", type="primary"):
        if not st.session_state.get("saved_questions_text"):
            st.error("Please save the questions first.")
            return
        if not answer_files:
            st.error("Please upload at least one student answer file.")
            return

        with st.spinner("Evaluating all uploaded answers..."):
            rows = []
            for uploaded_file in answer_files:
                answer_text = extract_text_from_upload(uploaded_file)
                result = grade_submission(
                    answer_text,
                    rubric_text,
                    index,
                    questions_text=st.session_state["saved_questions_text"],
                )
                rows.append(
                    {
                        "file": uploaded_file.name,
                        "total_score": result.get("total_score", 0),
                        "status": "Pass" if result.get("total_score", 0) >= 60 else "Needs review",
                        "flags": ", ".join(result.get("flags", [])) or "None",
                        "blocked": result.get("blocked", False),
                    }
                )
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True)
            if not summary_df.empty:
                st.bar_chart(summary_df.set_index("file")["total_score"])


def render_results(result: dict):
    if result.get("blocked"):
        st.error(result.get("reason", "Automatic grading stopped."))
        return

    total_score = result.get("total_score", 0)
    st.success(f"Total score: {total_score}/100")
    st.write(result.get("feedback", ""))
    if result.get("flags"):
        st.warning("Flags: " + ", ".join(result.get("flags", [])))

    for criterion in result.get("criteria", []):
        with st.expander(f"{criterion['criterion']} — {criterion['score']}/{criterion['max_marks']}"):
            st.write("**Justification:**", criterion.get("justification", ""))
            st.write("**Search query:**", criterion.get("search_query", ""))
            if criterion.get("book_quotes"):
                st.write("**Book quotes:**")
                for quote in criterion.get("book_quotes", []):
                    st.write("- " + str(quote))
            else:
                st.write("No direct quotes were captured.")
            if criterion.get("flags"):
                st.write("**Flags:**")
                for flag in criterion.get("flags", []):
                    st.write("- " + str(flag))


def main():
    st.title("AI Assignment Grader")
    st.caption("Upload the assignment questions and one or more student answer files to grade them against the book.")

    rubric_text = get_rubric_text()
    with st.spinner("Initializing the book index..."):
        index = get_index()

    render_questions_section()

    tab1, tab2 = st.tabs(["Single grading", "Bulk grading"])
    with tab1:
        render_single_grading(index, rubric_text)
    with tab2:
        render_bulk_grading(index, rubric_text)


if __name__ == "__main__":
    main()
